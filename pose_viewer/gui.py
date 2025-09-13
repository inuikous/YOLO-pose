"""pose_viewer 用の tkinter GUI（最小実装）。

責務:
 - Start/Stop ボタンを提供
 - 現在のフレームを表示（コントローラから ``after()`` で更新）
 - FPS と検出人数を表示
 - 入力フレームをウィジェットの表示領域にアスペクト比を維持してフィット
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from PIL import Image, ImageTk


class PoseViewerApp(tk.Tk):
    """姿勢ビューアアプリのメインウィンドウ。

    Args:
        display_w: 表示領域の幅（ピクセル）。
        display_h: 表示領域の高さ（ピクセル）。
    """

    def __init__(self, display_w: int = 960, display_h: int = 540):
        super().__init__()
        self.title("Pose Viewer (OpenVINO)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.minsize(640, 480)
        # フィードバックループ（画像→ウィジェット拡大→画像拡大...）を避けるための固定表示領域
        self.display_w = int(display_w)
        self.display_h = int(display_h)

        # 外部から設定されるコールバック参照
        self.on_start_cb = None
        self.on_stop_cb = None

        self._build_ui()
        self._img_tk: Optional[ImageTk.PhotoImage] = None

    # ---------------- UI -----------------
    def _build_ui(self):
        """ウィジェット群の構築を行います。"""
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=4)

        self.start_btn = ttk.Button(top, text="Start", command=self._click_start)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(top, text="Stop", command=self._click_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        self.fps_var = tk.StringVar(value="FPS: -")
        self.info_var = tk.StringVar(value="Persons: -")
        ttk.Label(top, textvariable=self.fps_var).pack(side=tk.LEFT, padx=12)
        ttk.Label(top, textvariable=self.info_var).pack(side=tk.LEFT, padx=12)

        # 子ウィジェットのサイズに合わせて親が拡大しない固定サイズコンテナ
        self.display_frame = tk.Frame(self, width=self.display_w, height=self.display_h, bg="#111")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        # 子（画像ラベル）のサイズでコンテナが変化しないようにする → フィードバック的な拡大を防止
        self.display_frame.pack_propagate(False)

        self.canvas = tk.Label(self.display_frame, text="No Frame", bg="#222", fg="#ccc")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ------------- Public Methods -------------
    def set_start_stop_callbacks(self, on_start, on_stop):
        """Start/Stop ボタンにコールバックを設定します。

        Args:
            on_start: Start ボタン押下時に呼ばれる関数。
            on_stop: Stop ボタン押下時に呼ばれる関数。
        """
        self.on_start_cb = on_start
        self.on_stop_cb = on_stop

    def update_frame(self, pil_image, persons: int, fps: float):
        """新しいフレームを表示し、指標を更新します。

        Args:
            pil_image: 表示対象の ``PIL.Image``。
            persons: 検出された人数。
            fps: 表示中のフレームレート。
        """
        # 現在のウィジェットサイズに画像をフィットさせる（アスペクト比維持）
        # 固定コンテナサイズを用いて自動拡大量のフィードバックループを防止
        avail_w = max(1, self.display_frame.winfo_width()) or self.display_w
        avail_h = max(1, self.display_frame.winfo_height()) or self.display_h
        img = self._fit_image(pil_image, avail_w, avail_h)

        # Convert PIL image to Tk
        self._img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self._img_tk)
        self.fps_var.set(f"FPS: {fps:.1f}")
        self.info_var.set(f"Persons: {persons}")

    def set_running_state(self, running: bool):
        """実行状態に応じてボタンの活性/非活性を切り替えます。"""
        if running:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    # ------------- Events -------------
    def _click_start(self):
        """Start ボタン押下時の内部ハンドラ。"""
        if self.on_start_cb:
            self.on_start_cb()

    def _click_stop(self):
        """Stop ボタン押下時の内部ハンドラ。"""
        if self.on_stop_cb:
            self.on_stop_cb()

    def _on_close(self):
        """ウィンドウクローズ時の処理。可能なら停止コールバックを呼ぶ。"""
        if self.on_stop_cb:
            try:
                self.on_stop_cb()
            except Exception:  # noqa
                pass
        self.destroy()

    # ------------- Helpers -------------
    @staticmethod
    def _fit_image(pil_img: Image, max_w: int, max_h: int) -> Image:
        """画像を指定サイズに収まるように高品質でリサイズします。

        Args:
            pil_img: 入力の ``PIL.Image``。
            max_w: 収めたい最大幅。
            max_h: 収めたい最大高さ。

        Returns:
            PIL.Image: リサイズ後の画像。
        """
        w, h = pil_img.size
        if w <= 0 or h <= 0:
            return pil_img
        scale = min(max_w / w, max_h / h)
        if scale <= 0:
            return pil_img
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        if new_w == w and new_h == h:
            return pil_img
        return pil_img.resize((new_w, new_h), Image.LANCZOS)
