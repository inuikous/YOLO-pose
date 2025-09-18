"""OpenVINO backend for TorchVision Keypoint R-CNN (IR model).

特徴:
- OpenVINO Runtime のみで IR (xml/bin) を読み込み推論
- 画像のリサイズは基本不要（動的形状モデル想定）。静的形状の場合はリサイズして逆スケール
- 前処理: BGR→RGB, float32/255, HWC→NCHW（TorchVision Transform/正規化はIR内部に含まれる想定）
- 後処理: 出力から boxes/scores/keypoints をヒューリスティックに抽出
  - 代表的な形状: boxes [N,4], scores [N], labels [N], keypoints [N,17,3] or [N,51]
  - 代替: heatmaps [N,17,H,W] / [N,H,W,17] を argmax で (x,y,score) に復元
- 出力座標が 0..1 正規化なら画像サイズへスケール、画素座標ならそのまま
- NMS は基本不要（RCNN 側で済みのことが多い）が、オプションで簡易 NMS を同梱

返却形式は `backend_openvino.OpenVinoPoseBackend` と互換:
list[{
  "bbox": (x1,y1,x2,y2) int,
  "keypoints": [(x,y,score), ...] float,
  "score": float,
  "class_id": int
}]
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from openvino.runtime import Core


class OpenVinoKeypointRCNNBackend:
    """TorchVision Keypoint R-CNN 用 OpenVINO バックエンド。"""

    # 画像正規化は行わない（IR内部のTransformに委譲）
    EPS = 1e-9

    def __init__(
        self,
        model_xml_path: str,
        device: str = "CPU",
        img_size: int = 640,
        conf_thres: float = 0.25,
        max_detections: int = 10,
        kpt_conf_thres: float = 0.20,
        nms_iou_thres: float = 0.45,
    ):
        self.model_xml_path = model_xml_path
        self.device = device
        self.conf_thres = float(conf_thres)
        self.kpt_conf_thres = float(kpt_conf_thres)
        self.max_detections = int(max_detections)
        self.nms_iou_thres = float(nms_iou_thres)

        core = Core()
        try:
            model = core.read_model(model=model_xml_path)
        except Exception as e:  # noqa
            raise RuntimeError(f"OpenVINOモデル(XML)の読み込みに失敗しました: {model_xml_path}\n{e}")
        try:
            self.compiled = core.compile_model(model=model, device_name=self.device)
        except Exception as e:  # noqa
            raise RuntimeError(f"OpenVINOモデルのコンパイルに失敗しました (device={self.device}):\n{e}")

        # 入力形状（動的対応）。静的なら H/W を保持
        self.in_h = None
        self.in_w = None
        ps = None
        try:
            ps = self.compiled.input(0).get_partial_shape()
        except Exception:
            try:
                ps = model.input(0).get_partial_shape()  # type: ignore[attr-defined]
            except Exception:
                ps = None

        def _try_extract_hw(partial_shape) -> tuple[int | None, int | None]:
            if partial_shape is None:
                return None, None
            # 試行: max→min の順で 4次元配列を取得
            for meth in ("get_max_shape", "get_min_shape"):
                try:
                    arr = getattr(partial_shape, meth)()
                    if isinstance(arr, (list, tuple)) and len(arr) == 4:
                        h = arr[2] if isinstance(arr[2], (int, np.integer)) and arr[2] > 0 else None
                        w = arr[3] if isinstance(arr[3], (int, np.integer)) and arr[3] > 0 else None
                        return h, w
                except Exception:
                    continue
            return None, None

        hh, ww = _try_extract_hw(ps)
        self.in_h = hh
        self.in_w = ww

        # 出力レイヤ参照
        self.output_layers = self.compiled.outputs

    # ---------- 前処理 ----------
    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """BGR → RGB/255 NCHW に変換。動的入力なら原寸、静的ならリサイズ。

        Returns:
            x: (1,3,H,W) float32
            meta: dict with keys 'scale_w', 'scale_h', 'orig_w', 'orig_h', 'resized'
        """
        h, w = bgr.shape[:2]
        resized = False
        if self.in_h is not None and self.in_w is not None and (self.in_h != h or self.in_w != w):
            bgr = cv2.resize(bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
            resized = True
        hh, ww = bgr.shape[:2]
        rgb = bgr[..., ::-1]
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = x[None, ...]  # -> NCHW
        meta = {
            "scale_w": (ww / w),
            "scale_h": (hh / h),
            "orig_w": w,
            "orig_h": h,
            "resized": resized,
        }
        return x, meta

    # ---------- 出力抽出ユーティリティ ----------
    @staticmethod
    def _squeeze_leading_ones(a: np.ndarray) -> np.ndarray:
        while a.ndim > 0 and a.shape[0] == 1:
            a = a.reshape(a.shape[1:])
        return a

    @staticmethod
    def _coalesce(*values):
        for v in values:
            if v is not None:
                return v
        return None

    @staticmethod
    def _clip_boxes_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _decode_heatmaps(hm: np.ndarray) -> np.ndarray:
        """heatmaps -> keypoints [N,17,3] with (x,y,score)."""
        # accept [N,17,H,W] or [N,H,W,17]
        if hm.ndim == 4 and hm.shape[1] == 17:
            hm = hm  # [N,17,H,W]
        elif hm.ndim == 4 and hm.shape[-1] == 17:
            hm = np.transpose(hm, (0, 3, 1, 2))
        else:
            return None
        N, K, H, W = hm.shape
        flat = hm.reshape(N, K, H * W)
        idx = np.argmax(flat, axis=2)
        scores = np.max(flat, axis=2)
        ys = (idx // W).astype(np.float32)
        xs = (idx % W).astype(np.float32)
        # scale to heatmap size; caller will rescale if normalized
        kpts = np.stack([xs, ys, scores], axis=2)  # [N,K,3]
        return kpts

    def _pick_outputs(self, outputs: Dict[str, np.ndarray]) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """boxes [N,4], scores [N], keypoints [N,17,3] を抽出（可能な範囲で）。
        keypoint_scores [N,17] が存在する場合は keypoints[...,2] として統合。"""
        boxes = None
        scores = None
        kpts = None
        kp_scores = None

        # squeeze and collect
        outs = {name: self._squeeze_leading_ones(np.array(val)) for name, val in outputs.items()}

        # name-based quick picks
        for name, a in outs.items():
            n = name.lower()
            if boxes is None and a.ndim == 2 and a.shape[-1] in (4, 5) and ("box" in n or "boxes" in n or "dets" in n):
                boxes = a[:, :4].astype(np.float32)
                if a.shape[-1] == 5 and scores is None:
                    scores = a[:, 4].astype(np.float32)
            if scores is None and a.ndim >= 1 and ("score" in n or "scores" in n):
                scores = a.reshape(-1).astype(np.float32)
            if kpts is None and ("keypoint" in n or "kpts" in n or "keypoints" in n):
                if a.ndim == 3 and a.shape[-1] == 3:
                    kpts = a.astype(np.float32)
                elif a.ndim == 2 and a.shape[-1] % 3 == 0:
                    K = a.shape[-1] // 3
                    kpts = a.reshape(a.shape[0], K, 3).astype(np.float32)
            # keypoint_scores 専用
            if kp_scores is None and ("keypoint_scores" in n or ("keypoint" in n and "score" in n)):
                if a.ndim == 2 and (a.shape[-1] == 17 or a.shape[-1] == 18 or a.shape[-1] <= 32):
                    kp_scores = a.astype(np.float32)

        # shape-based fallbacks
        if boxes is None:
            cand = [a for a in outs.values() if a.ndim == 2 and a.shape[-1] in (4, 5)]
            if cand:
                a = cand[0]
                boxes = a[:, :4].astype(np.float32)
                if a.shape[-1] == 5 and scores is None:
                    scores = a[:, 4].astype(np.float32)
        if scores is None:
            cand = [a for a in outs.values() if a.ndim in (1, 2) and min(a.shape) == 1]
            if cand:
                scores = cand[0].reshape(-1).astype(np.float32)
            if kpts is None:
                # heatmaps?
                for a in outs.values():
                    k = self._decode_heatmaps(a)
                    if k is not None:
                        kpts = k.astype(np.float32)
                        break
                if kpts is None:
                    cand = [
                        a for a in outs.values()
                        if (a.ndim == 3 and a.shape[-1] in (2, 3))
                        or (a.ndim == 2 and (a.shape[-1] % 3 == 0 or a.shape[-1] % 2 == 0))
                    ]
                    if cand:
                        a = cand[0]
                        if a.ndim == 3:
                            if a.shape[-1] == 3:
                                kpts = a.astype(np.float32)
                            else:  # last dim==2 → add score=1.0
                                ones = np.ones((a.shape[0], a.shape[1], 1), dtype=a.dtype)
                                kpts = np.concatenate([a.astype(np.float32), ones.astype(np.float32)], axis=2)
                        else:
                            if a.shape[-1] % 3 == 0:
                                K = a.shape[-1] // 3
                                kpts = a.reshape(a.shape[0], K, 3).astype(np.float32)
                            else:  # assume XY only
                                K = a.shape[-1] // 2
                                xy = a.reshape(a.shape[0], K, 2).astype(np.float32)
                                ones = np.ones((a.shape[0], K, 1), dtype=np.float32)
                                kpts = np.concatenate([xy, ones], axis=2)

        # keypoint_scores を統合
        if kpts is not None and kp_scores is not None:
            try:
                ks = kp_scores.reshape(kpts.shape[0], kpts.shape[1]).astype(np.float32)
                # kpts[...,2] に上書き
                kpts = kpts.copy()
                kpts[..., 2] = ks
            except Exception:
                pass

        return boxes, scores, kpts

    # ---------- 推論 ----------
    def infer(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        x, meta = self._preprocess(bgr_image)
        H0, W0 = meta["orig_h"], meta["orig_w"]
        Hx, Wx = x.shape[2], x.shape[3]

        ov_res = self.compiled(x)
        # name -> ndarray（ポート順に安全に取得）
        outputs: Dict[str, np.ndarray] = {}
        for port in self.output_layers:
            name = port.get_any_name() or ""
            try:
                outputs[name or f"out_{port.index}"] = np.array(ov_res[port])  # type: ignore[attr-defined]
            except Exception:
                # フォールバック: 名前のみで取得
                try:
                    outputs[name or "out"] = np.array(ov_res[name])
                except Exception:
                    continue

        boxes, scores, kpts = self._pick_outputs(outputs)

        if boxes is None or scores is None:
            # 何も抽出できない場合は空
            return []

        # 一貫性調整（N を合わせる）
        N = min(len(boxes), len(scores))
        if kpts is not None:
            N = min(N, len(kpts))
        boxes = boxes[:N]
        scores = scores[:N]
        if kpts is not None:
            kpts = kpts[:N]

        # 正規化検出（0..1 っぽいか）
        def _looks_normalized(arr: np.ndarray) -> bool:
            a = arr.reshape(-1)
            if a.size == 0:
                return False
            m = float(np.nanmax(a))
            return 0.0 <= float(np.nanmin(a)) <= 1.2 and m <= 1.2

        # boxes: xyxy or xywh? → 右下が左上未満の値が混じると xywh の可能性
        xyxy = boxes.copy()
        # 粗判定: 中央が大きく負 or x2<x1 多数なら xywh と仮定
        xyxy_bad = np.mean(xyxy[:, 2] < xyxy[:, 0]) + np.mean(xyxy[:, 3] < xyxy[:, 1])
        if xyxy_bad > 0.5:
            xyxy = self._xywh_to_xyxy(boxes)

        # 正規化なら画像サイズにスケール
        if _looks_normalized(xyxy):
            xyxy[:, [0, 2]] *= Wx
            xyxy[:, [1, 3]] *= Hx

        # 静的入力でリサイズした場合は元画像へ逆スケール
        if meta["resized"] and (self.in_w and self.in_h):
            sx = W0 / max(1e-6, float(self.in_w))
            sy = H0 / max(1e-6, float(self.in_h))
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy

        xyxy = self._clip_boxes_xyxy(xyxy, W0, H0)

        # keypoints
        kpts_list: List[List[Tuple[float, float, float]]] | None = None
        if kpts is not None:
            kk = kpts.astype(np.float32)
            if _looks_normalized(kk[..., :2]):
                kk[..., 0] *= Wx
                kk[..., 1] *= Hx
            if meta["resized"] and (self.in_w and self.in_h):
                kk[..., 0] *= (W0 / max(1e-6, float(self.in_w)))
                kk[..., 1] *= (H0 / max(1e-6, float(self.in_h)))
            # 画素境界にクリップ
            kk[..., 0] = np.clip(kk[..., 0], 0, W0 - 1)
            kk[..., 1] = np.clip(kk[..., 1], 0, H0 - 1)
            # (N, K, 3) -> Python list of tuples
            kpts_list = [[(float(x), float(y), float(v)) for (x, y, v) in kp] for kp in kk]

        # スコアしきい値・上位K
        order = np.argsort(-scores)
        kept: List[int] = []
        for idx in order:
            if float(scores[idx]) < self.conf_thres:
                continue
            kept.append(int(idx))
            if 0 < self.max_detections <= len(kept):
                break

        dets: List[Dict[str, Any]] = []
        for i in kept:
            bbox = tuple(int(round(v)) for v in xyxy[i])
            keypoints = kpts_list[i] if kpts_list is not None else []
            dets.append({
                "bbox": bbox,
                "keypoints": keypoints,
                "score": float(scores[i]),
                "class_id": 0,
            })

        return dets

    def close(self):
        pass
