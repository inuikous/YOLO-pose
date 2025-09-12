"""Entry point for pose_viewer.

Implements minimal pipeline:
  - Load config
  - Start tkinter GUI
  - Start/Stop capture thread reading webcam frames
  - Dummy backend inference (OpenVinoPoseBackend returning synthetic data)
  - Draw skeleton using pose_draw utilities

This is Step A (dummy inference). Will be replaced with real OpenVINO decoding later.
"""

from __future__ import annotations

import threading
import time
import queue
import yaml
from pathlib import Path

import cv2
from PIL import Image

from backend_openvino import OpenVinoPoseBackend
from gui import PoseViewerApp
from pose_draw import draw_pose


class CaptureController:
    def __init__(self, app: PoseViewerApp, config_path: str = "config.yaml"):
        self.app = app
        self.config_path = config_path
        self.cfg = self._load_config()
        self.running = False
        self.thread: threading.Thread | None = None
        self.backend = OpenVinoPoseBackend(
            model_xml_path=self.cfg["model"]["path"],
            device=self.cfg["model"].get("device", "CPU"),
            img_size=self.cfg["model"].get("img_size", 640),
        )
        self.cap = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.last_fps = 0.0
        self.skip = int(self.cfg.get("performance", {}).get("skip", 0))
        self.frame_count = 0

    # -------------- Config --------------
    def _load_config(self):
        path = Path(__file__).parent / self.config_path
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # -------------- Lifecycle --------------
    def start(self):
        if self.running:
            return
        source = self.cfg["input"].get("source", 0)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.backend.close()

    # -------------- Main Loop --------------
    def _loop(self):
        prev_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            self.frame_count += 1
            if self.skip and (self.frame_count % (self.skip + 1) != 1):
                continue

            detections = self.backend.infer(frame)
            # Draw each detection
            persons = 0
            for det in detections:
                persons += 1
                frame = draw_pose(
                    frame,
                    det["keypoints"],
                    skeleton=[
                        (16, 14), (14, 12), (17, 15), (15, 13), (12, 6), (13, 6),
                        (6, 7), (7, 8), (8, 9), (9, 10), (6, 2), (2, 3), (3, 4),
                        (6, 1), (1, 5), (5, 11), (11, 12)
                    ],
                )

            # FPS calc
            now = time.time()
            dt = now - prev_time
            if dt > 0:
                self.last_fps = 1.0 / dt
            prev_time = now

            # Convert to PIL for Tk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Update GUI (in main thread) using after
            self.app.after(0, self.app.update_frame, pil_img, persons, self.last_fps)

        # End loop


def main():
    app = PoseViewerApp()
    controller = CaptureController(app)

    def on_start():
        try:
            controller.start()
            app.set_running_state(True)
        except Exception as e:  # noqa
            from tkinter import messagebox
            messagebox.showerror("Error", str(e))

    def on_stop():
        controller.stop()
        app.set_running_state(False)

    app.set_start_stop_callbacks(on_start, on_stop)
    app.mainloop()


if __name__ == "__main__":
    main()
