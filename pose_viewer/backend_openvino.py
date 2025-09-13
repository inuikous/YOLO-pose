"""OpenVINO バックエンド（IR を OpenVINO ランタイムのみで推論）。

最小構成で以下を行います:
    1. ``openvino.runtime.Core`` を用いて XML（および BIN）からモデルを読み込み/コンパイル
    2. 入力フレームの前処理（レターボックス、RGB 変換、NCHW、0-1 正規化）
    3. 推論実行と YOLO Pose っぽい出力の簡易デコード（汎用的な形状推定）

注: Ultralytics ライブラリは使用せず、OpenVINO のみで実行します。モデルの出力レイアウトはエクスポート条件で差異があり得るため、
可能な範囲で柔軟に解釈します（典型: [N, 56] = [x, y, w, h, conf, 17*3(kpts)]）。
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from openvino.runtime import Core


class OpenVinoPoseBackend:
    """OpenVINO を用いたポーズ推論バックエンド（ダミー）。

    Args:
        model_xml_path: OpenVINO IR の XML ファイルパス。
        device: 推論デバイス名（例: ``"CPU"``、``"AUTO"`` など）。
        img_size: 推論時の入力解像度（正方形想定）。
    """

    def __init__(self, model_xml_path: str, device: str = "CPU", img_size: int = 640):
        self.model_xml_path = model_xml_path
        self.device = device
        self.img_size = int(img_size)
        self.conf_thres = 0.25  # 簡易しきい値（必要に応じて設定化）

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
        # レターボックス + RGB + NCHW + float32/255
        img_lb, r, (dw, dh) = self._letterbox(bgr, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, 0)        # CHW -> NCHW
        return x, r, (dw, dh)

    def infer(self, bgr_image) -> List[Dict[str, Any]]:
        """OpenVINO で実推論を行い、検出結果を返します。

        Args:
            bgr_image: BGR 配列（``numpy.ndarray``）。

        Returns:
            list[dict]: 検出結果のリスト。各要素は以下のキーを持ちます。

            - ``bbox``: ``(x1, y1, x2, y2)`` の境界ボックス（画素座標）
            - ``keypoints``: ``[(x, y, score), ...]`` 形式のキーポイント（COCO 17点相当を想定）
            - ``score``: 検出スコア
            - ``class_id``: クラス ID（人: 0 を想定）
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
        dets: List[Dict[str, Any]] = []

        # C からキーポイント数と項目の推定
        # 想定パターン: [x, y, w, h, conf, kpts(17*3)] -> 56
        # もしくは conf 無し: 55 (= 4 + 17*3)
        has_conf = False
        kpt_start = 4
        kpt_num = 17
        if (C - 5) % 3 == 0 and (C - 5) // 3 >= 5:  # 粗い安全策
            has_conf = True
            kpt_start = 5
            kpt_num = (C - 5) // 3
        elif (C - 4) % 3 == 0 and (C - 4) // 3 >= 5:
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
            # xyxy かどうかの簡易判定（右下 > 左上）
            if w_or_x2 > x and h_or_y2 > y:
                x1_l = w_or_x2
                y1_l = h_or_y2
                # 実は xyxy ではなく x2,y2 を受け取ったとみなす
                x1i, y1i, x2i, y2i = x, y, w_or_x2, h_or_y2
            else:
                # xywh を xyxy へ変換
                x1i = x - w_or_x2 / 2
                y1i = y - h_or_y2 / 2
                x2i = x + w_or_x2 / 2
                y2i = y + h_or_y2 / 2

            # キーポイント復元
            kpts = []
            for k in range(kpt_num):
                base = kpt_start + k * 3
                if base + 2 >= C:
                    break
                kx, ky, kc = float(row[base]), float(row[base + 1]), float(row[base + 2])
                # レターボックス逆写像
                ox = (kx - dw) / (r + 1e-9)
                oy = (ky - dh) / (r + 1e-9)
                kpts.append((ox, oy, kc))

            # bbox も逆写像
            ox1 = (x1i - dw) / (r + 1e-9)
            oy1 = (y1i - dh) / (r + 1e-9)
            ox2 = (x2i - dw) / (r + 1e-9)
            oy2 = (y2i - dh) / (r + 1e-9)

            dets.append({
                "bbox": (int(ox1), int(oy1), int(ox2), int(oy2)),
                "keypoints": kpts,
                "score": score,
                "class_id": 0,
            })

        return dets

    def close(self):
        """リソース解放のためのプレースホルダ関数。"""
        pass
