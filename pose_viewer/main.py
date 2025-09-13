"""pose_viewer のエントリポイント。

最小パイプラインを実装:
    - 設定ファイルの読み込み
    - tkinter GUI の起動
    - Web カメラからのフレーム取得用スレッドの Start/Stop
    - OpenVINO バックエンドでの推論
    - `pose_draw` を使ってスケルトン描画
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


# -------------- Helpers --------------

def get_base_dir() -> Path:
    """このスクリプトが存在するディレクトリを返します。"""
    return Path(__file__).resolve().parent


def load_config(base_dir: Path, config_filename: str = "config.yaml") -> dict:
    """YAML 設定を読み込み辞書で返します。"""
    cfg_path = base_dir / config_filename
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CaptureController:
    """カメラキャプチャと推論、GUI 更新を統括するコントローラ。"""

    def __init__(
        self,
        app: PoseViewerApp,
        config_path: str = "config.yaml",
        cfg: dict | None = None,
        base_dir: Path | None = None,
    ):
        self.app = app
        self.config_path = config_path
        self.base_dir = base_dir or get_base_dir()
        self.cfg = cfg if cfg is not None else load_config(self.base_dir, self.config_path)
        self.running = False
        self.thread: threading.Thread | None = None

        # モデルパスを base_dir 基準で解決（CWD非依存）
        model_path_cfg = str(self.cfg["model"]["path"]) if isinstance(self.cfg["model"].get("path"), str) else self.cfg["model"]["path"]
        model_path = Path(model_path_cfg)
        if not model_path.is_absolute():
            model_path = (self.base_dir / model_path).resolve()
        if not model_path.exists():
            raise RuntimeError(f"Model XML not found: {model_path}")

        self.backend = OpenVinoPoseBackend(
            model_xml_path=str(model_path),
            device=self.cfg["model"].get("device", "CPU"),
            img_size=self.cfg["model"].get("img_size", 640),
        )
        self.cap = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.last_fps = 0.0
        self.skip = int(self.cfg.get("performance", {}).get("skip", 0))
        self.frame_count = 0

    def start(self):
        """キャプチャスレッドを開始し、必要ならキャプチャ設定を適用します。"""
        if self.running:
            return
        source = self.cfg["input"].get("source", 0)
        self.cap = cv2.VideoCapture(source)
        # Apply capture settings if provided
        cap_cfg = self.cfg["input"].get("capture", {})
        cw = int(cap_cfg.get("width", 0) or 0)
        ch = int(cap_cfg.get("height", 0) or 0)
        cfps = int(cap_cfg.get("fps", 0) or 0)
        if cw > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
        if ch > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
        if cfps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, cfps)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {source}")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """キャプチャスレッドを停止し、リソースを解放します。"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.backend.close()

    def _loop(self):
        """キャプチャ→推論→描画→GUI 更新のループ本体。"""
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
    """アプリケーションのメイン関数。設定読み込み→GUI 起動を行います。"""
    # 設定を読み込んで GUI に表示サイズを渡す
    base_dir = get_base_dir()
    cfg = load_config(base_dir)

    disp_cfg = cfg.get("display", {})
    app = PoseViewerApp(
        display_w=int(disp_cfg.get("width", 960)),
        display_h=int(disp_cfg.get("height", 540)),
    )
    controller = CaptureController(app, cfg=cfg, base_dir=base_dir)

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
