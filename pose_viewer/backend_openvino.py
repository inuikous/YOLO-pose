"""OpenVINO バックエンド（IR を OpenVINO ランタイムのみで推論）。

概要:
        - OpenVINO IR（``.xml`` と ``.bin``）を読み込み、CPU/GPU 等で推論
        - 前処理: レターボックス、BGR→RGB、``float32/255``、``NCHW``
        - 後処理: 出力形状の転置検出、0..1 正規化座標のスケーリング、
            YOLO系前提の ``xywh``（中心+幅高）→``xyxy`` 変換、レターボックス逆写像、
            スコアしきい値、NMS（IoUベースの貪欲法）により重複抑制、
            ``max_detections`` までに制限

注意:
        - Ultralytics などの外部推論ライブラリは不要（OpenVINO のみ）
        - モデル出力レイアウトはエクスポート条件で差異があるため、
            代表例（``[N, 56]`` もしくは ``[56, N]``）に対してヒューリスティックに解釈する
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from openvino.runtime import Core


class OpenVinoPoseBackend:
    """OpenVINO を用いたポーズ推論バックエンド。

    Args:
        model_xml_path: OpenVINO IR の XML ファイルパス。
        device: 推論デバイス名（例: ``"CPU"``, ``"AUTO"`` など）。
        img_size: モデル入力の想定解像度（正方形）※実入力はレターボックスで合わせる。
        conf_thres: 検出を残す最低スコア。
    max_detections: 返却する最大検出数。
    nms_iou_thres: NMS の IoU しきい値（例: 0.45）。

    備考:
        - 入力テンソル形状は ``[1, 3, H, W]`` を期待。
        - 出力は ``[N, C]`` または ``[C, N]`` を想定し、後者は転置して扱う。
    """

    # クラス定数（ハードコード値の集約）
    MIN_KPTS_FOR_VALID_LAYOUT = 5  # レイアウト推定の最低KP数
    NORM_RANGE_MAX = 1.2           # 正規化と見なす上限
    KPTS_NORM_CHECKS = 5           # 正規化判定で見る点数の上限
    EPS = 1e-9                     # 数値安定化用の微小値

    def __init__(self, model_xml_path: str, device: str = "CPU", img_size: int = 640, conf_thres: float = 0.25, max_detections: int = 10, kpt_conf_thres: float = 0.20, nms_iou_thres: float = 0.45):
        self.model_xml_path = model_xml_path
        self.device = device
        self.img_size = int(img_size)
        # bbox用（検出）とkpt用（描画）のしきい値
        self.conf_thres = float(conf_thres)
        self.kpt_conf_thres = float(kpt_conf_thres)
        # NMS と最終的な max_detections 制限に使用
        self.max_detections = int(max_detections)
        self.nms_iou_thres = float(nms_iou_thres)

        # OpenVINO モデルの読み込みとコンパイル
        core = Core()
        try:
            model = core.read_model(model=model_xml_path)
        except Exception as e:  # noqa
            raise RuntimeError(f"OpenVINOモデル(XML)の読み込みに失敗しました: {model_xml_path}\n{e}")
        try:
            self.compiled = core.compile_model(model=model, device_name=self.device)
        except Exception as e:  # noqa
            raise RuntimeError(f"OpenVINOモデルのコンパイルに失敗しました (device={self.device}):\n{e}")

        # 入出力レイヤと入力サイズ
        self.input_layer = self.compiled.input(0)
        in_shape = list(self.input_layer.shape)
        # 期待形状: [1, 3, H, W]
        if len(in_shape) != 4:
            raise RuntimeError(f"想定外の入力形状: {in_shape}. 期待: [N, C, H, W]")
        self.input_h = int(in_shape[2])
        self.input_w = int(in_shape[3])
        self.output_layers = self.compiled.outputs

    # ---------------- 前処理 ----------------
    @staticmethod
    def _letterbox(img: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """レターボックス: アスペクト比を保ちつつ new_shape に収め、上下左右を埋める。

        Args:
            img: BGR 画像。
            new_shape: (w, h) へ収める。
            color: パディング色。

        Returns:
            tuple: (パディング後画像, 比率r, 余白(dw, dh))
        """
        h, w = img.shape[:2]
        new_w, new_h = int(new_shape[0]), int(new_shape[1])
        r = min(new_w / w, new_h / h)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw = new_w - new_unpad[0]
        dh = new_h - new_unpad[1]
        dw /= 2
        dh /= 2

        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """推論前の最小前処理を行う。

        手順:
            1) レターボックスで ``(input_w, input_h)`` に収める
            2) BGR→RGB 変換
            3) ``float32/255`` 正規化
            4) ``HWC→CHW`` → バッチ次元付与で ``NCHW``

        Args:
            bgr: 入力フレーム（BGR）。

        Returns:
            tuple[numpy.ndarray, float, tuple]:
                - 推論入力（NCHW, float32）
                - レターボックスのスケール係数 ``r``
                - 余白 ``(dw, dh)``
        """
        # レターボックス + RGB + NCHW + float32/255
        img_lb, r, (dw, dh) = self._letterbox(bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, 0)        # CHW -> NCHW
        return x, r, (dw, dh)

    def infer(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        """OpenVINO で推論し、検出リストを返す（最小構成）。

        処理の流れ:
            - 前処理（レターボックス等）→ 推論 → 出力解釈 → 座標復元 → 上位K選抜

        出力の解釈ポリシー:
            - 形状が ``[C, N]`` と判断できれば ``[N, C]`` へ転置
            - YOLO系の標準として bbox は ``xywh``（中心+幅高）を想定して ``xyxy`` に変換
            - 座標が 0..1 の正規化ならモデル入力サイズにスケールした後、
              レターボックスの余白/スケールを逆変換して元画像座標に戻す
                        - スコアしきい値で間引き後、NMS（IoU 閾値）で重複抑制し、
                            降順で ``max_detections`` 件に制限

        Args:
            bgr_image: 入力フレーム（BGR, ``numpy.ndarray``）。

        Returns:
            list[dict]: 検出結果のリスト。各要素は以下のキーを持つ。
                - ``bbox``: ``(x1, y1, x2, y2)``（int, 画素座標）
                - ``keypoints``: ``[(x, y, score), ...]``（float, 画素座標）
                - ``score``: 検出スコア（float）
                - ``class_id``: クラス ID（int。人クラス 0 を想定）
        """
        img_input, r, (dw, dh) = self._preprocess(bgr_image)

        # 推論
        outputs = self.compiled([img_input])
        # 出力テンソルの取り出し（最初の出力を主に使用）
        if not outputs:
            return []
        out = outputs[0]
        arr = np.array(out)

        # 期待形状: [1, N, C] or [N, C]
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            # 想定外の場合は空
            return []

        N, C = arr.shape
        # 出力が [C, N] 形式（例: [56, 8400]）の場合に備え、
        # 「チャネル数がYOLO Poseとして妥当か」を指標に転置を検討
        def _chan_plausible(v: int) -> bool:
            return ((v - 5) % 3 == 0 and (v - 5) // 3 >= 5) or ((v - 4) % 3 == 0 and (v - 4) // 3 >= 5)
        if _chan_plausible(N) and not _chan_plausible(C):
            arr = arr.T
            N, C = arr.shape
        dets: List[Dict[str, Any]] = []

        # C からキーポイント数と項目の推定
        # 代表例: [x, y, w, h, conf, kpts(17*3)] -> 56
        # conf 無しのケース: 55 (= 4 + 17*3)
        has_conf = False
        kpt_start = 4
        kpt_num = 17
        if (C - 5) % 3 == 0 and (C - 5) // 3 >= self.MIN_KPTS_FOR_VALID_LAYOUT:  # 粗い安全策
            has_conf = True
            kpt_start = 5
            kpt_num = (C - 5) // 3
        elif (C - 4) % 3 == 0 and (C - 4) // 3 >= self.MIN_KPTS_FOR_VALID_LAYOUT:
            has_conf = False
            kpt_start = 4
            kpt_num = (C - 4) // 3

        # 1件ずつ復元
        for i in range(N):
            row = arr[i]
            # スコア判定（conf がない場合は平均キーポイント信頼度 or 1.0）
            score = float(row[4]) if has_conf else 1.0
            if has_conf and score < self.conf_thres:
                continue

            # bbox: xywh or xyxy 推定
            x, y, w_or_x2, h_or_y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            # 出力が 0..1 正規化の場合に備えて簡易判定し、入力解像度へスケール
            if 0.0 <= x <= self.NORM_RANGE_MAX and 0.0 <= y <= self.NORM_RANGE_MAX and 0.0 <= w_or_x2 <= self.NORM_RANGE_MAX and 0.0 <= h_or_y2 <= self.NORM_RANGE_MAX:
                x *= self.input_w
                y *= self.input_h
                w_or_x2 *= self.input_w
                h_or_y2 *= self.input_h
            # YOLO系の標準として bbox は [x, y, w, h]（中心座標+幅高）想定とし、xyxyへ変換
            x1i = x - w_or_x2 / 2
            y1i = y - h_or_y2 / 2
            x2i = x + w_or_x2 / 2
            y2i = y + h_or_y2 / 2

            # キーポイント復元
            kpts = []
            # キーポイントが 0..1 正規化かを事前にざっくり判定
            norm_votes = 0
            checks = min(self.KPTS_NORM_CHECKS, kpt_num)
            for k in range(checks):
                base = kpt_start + k * 3
                if base + 1 >= C:
                    break
                if 0.0 <= row[base] <= self.NORM_RANGE_MAX and 0.0 <= row[base + 1] <= self.NORM_RANGE_MAX:
                    norm_votes += 1
            kpts_normed = (checks > 0 and norm_votes >= max(2, checks - 1))

            for k in range(kpt_num):
                base = kpt_start + k * 3
                if base + 2 >= C:
                    break
                kx, ky, kc = float(row[base]), float(row[base + 1]), float(row[base + 2])
                if kpts_normed:
                    kx *= self.input_w
                    ky *= self.input_h
                # レターボックス逆写像
                ox = (kx - dw) / (r + self.EPS)
                oy = (ky - dh) / (r + self.EPS)
                # インデックス整合のため、ここでは全点を保持（描画側でスコア判定）
                kpts.append((ox, oy, kc))

            # bbox も逆写像
            ox1 = (x1i - dw) / (r + self.EPS)
            oy1 = (y1i - dh) / (r + self.EPS)
            ox2 = (x2i - dw) / (r + self.EPS)
            oy2 = (y2i - dh) / (r + self.EPS)

            dets.append({
                "bbox": (int(ox1), int(oy1), int(ox2), int(oy2)),
                "keypoints": kpts,
                "score": score,
                "class_id": 0,
            })

        # NMS（IoUベースの貪欲法）で重複抑制
        if dets:
            dets = self._nms_greedy(dets, self.nms_iou_thres, self.max_detections)

        return dets

    def close(self):
        """リソース解放のためのプレースホルダ関数。"""
        pass

    # ---------------- NMS（Greedy） ----------------
    @staticmethod
    def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
        """2つの矩形（xyxy）の IoU を返す。"""
        # 交差領域
        xx1 = max(a[0], b[0])
        yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2])
        yy2 = min(a[3], b[3])
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h
        # 各面積
        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def _nms_greedy(self, dets: List[Dict[str, Any]], iou_thres: float, max_det: int) -> List[Dict[str, Any]]:
        """スコア降順の貪欲 NMS を適用して重複検出を抑制する。

        Args:
            dets: 検出結果のリスト（`bbox`, `score`, `keypoints` を含む）。
            iou_thres: IoU しきい値（これを超える重複は除外）。
            max_det: 返却する最大数（0 以下なら制限なし）。

        Returns:
            NMS 後の検出リスト。
        """
        if not dets:
            return dets
        # スコア降順で並べ替え
        order = sorted(range(len(dets)), key=lambda i: dets[i].get("score", 0.0), reverse=True)
        kept: List[int] = []
        for idx in order:
            candidate = dets[idx]
            cand_box = np.array(candidate["bbox"], dtype=np.float32)
            suppress = False
            for kept_idx in kept:
                kept_box = np.array(dets[kept_idx]["bbox"], dtype=np.float32)
                if self._iou_xyxy(cand_box, kept_box) > iou_thres:
                    suppress = True
                    break
            if not suppress:
                kept.append(idx)
                if max_det > 0 and len(kept) >= max_det:
                    break
        return [dets[i] for i in kept]
