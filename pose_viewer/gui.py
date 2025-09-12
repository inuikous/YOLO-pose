"""tkinter GUI for pose_viewer (minimal initial implementation).

Responsibilities:
 - Provide Start/Stop buttons
 - Display current frame (updated via controller outside mainloop using after())
 - Show FPS and number of persons detected
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from PIL import Image, ImageTk


class PoseViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pose Viewer (OpenVINO)")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # State refs set externally
        self.on_start_cb = None
        self.on_stop_cb = None

        self._build_ui()
        self._img_tk: Optional[ImageTk.PhotoImage] = None

    # ---------------- UI -----------------
    def _build_ui(self):
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

        self.canvas = tk.Label(self, text="No Frame", bg="#222", fg="#ccc", width=80, height=25)
        self.canvas.pack(padx=8, pady=8)

    # ------------- Public Methods -------------
    def set_start_stop_callbacks(self, on_start, on_stop):
        self.on_start_cb = on_start
        self.on_stop_cb = on_stop

    def update_frame(self, pil_image, persons: int, fps: float):
        # Convert PIL image to Tk
        self._img_tk = ImageTk.PhotoImage(pil_image)
        self.canvas.configure(image=self._img_tk)
        self.fps_var.set(f"FPS: {fps:.1f}")
        self.info_var.set(f"Persons: {persons}")

    def set_running_state(self, running: bool):
        if running:
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    # ------------- Events -------------
    def _click_start(self):
        if self.on_start_cb:
            self.on_start_cb()

    def _click_stop(self):
        if self.on_stop_cb:
            self.on_stop_cb()

    def _on_close(self):
        if self.on_stop_cb:
            try:
                self.on_stop_cb()
            except Exception:  # noqa
                pass
        self.destroy()
