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
from backend_keypointrcnn_openvino import OpenVinoKeypointRCNNBackend
from gui import PoseViewerApp
from pose_draw import draw_pose
from settings import load_settings


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

        # Backend selection: YOLO-style vs Keypoint RCNN
        backend_kind = str(self.cfg["model"].get("backend", "auto")).lower()
        model_name = model_path.name.lower()
        use_rcnn = False
        if backend_kind == "rcnn":
            use_rcnn = True
        elif backend_kind == "yolo":
            use_rcnn = False
        else:
            # auto: pick by filename hint
            if any(k in model_name for k in ["keypointrcnn", "kprcnn", "rcnn"]):
                use_rcnn = True
            else:
                use_rcnn = False

        if use_rcnn:
            self.backend = OpenVinoKeypointRCNNBackend(
                model_xml_path=str(model_path),
                device=self.cfg["model"].get("device", "CPU"),
                img_size=self.cfg["model"].get("img_size", 640),
                conf_thres=float(self.cfg.get("postprocess", {}).get("det_confidence", 0.25)),
                max_detections=int(self.cfg.get("postprocess", {}).get("max_detections", 10)),
                kpt_conf_thres=float(self.cfg.get("postprocess", {}).get("kpt_confidence", 0.20)),
                nms_iou_thres=float(self.cfg.get("postprocess", {}).get("nms_iou", 0.45)),
            )
        else:
            self.backend = OpenVinoPoseBackend(
                model_xml_path=str(model_path),
                device=self.cfg["model"].get("device", "CPU"),
                img_size=self.cfg["model"].get("img_size", 640),
                conf_thres=float(self.cfg.get("postprocess", {}).get("det_confidence", 0.25)),
                max_detections=int(self.cfg.get("postprocess", {}).get("max_detections", 10)),
                kpt_conf_thres=float(self.cfg.get("postprocess", {}).get("kpt_confidence", 0.20)),
                nms_iou_thres=float(self.cfg.get("postprocess", {}).get("nms_iou", 0.45)),
            )
        # 以降、max_detections は backend 生成時に設定済み
        self.cap = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.last_fps = 0.0
        self.skip = int(self.cfg.get("performance", {}).get("skip", 0))
        self.frame_count = 0

    def start(self):
        """キャプチャスレッドを開始し、必要ならキャプチャ設定を適用します。"""
        if self.running:
            return
        source_cfg = self.cfg["input"].get("source", 0)
        # 入力ソース: 数値/数値文字列→int、ファイル/URL 文字列→パス解決
        source = source_cfg
        if isinstance(source_cfg, str):
            src_str = source_cfg.strip()
            if src_str.isdigit():
                source = int(src_str)
            elif src_str.lower().startswith(("rtsp://", "http://", "https://")):
                source = src_str
            else:
                # 相対パスは base_dir 基準で解決
                p = Path(src_str)
                if not p.is_absolute():
                    p = (self.base_dir / p).resolve()
                source = str(p)
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
            draw_cfg = self.cfg.get("draw", {})
            skel_on = bool(draw_cfg.get("skeleton", True))
            kp_radius = int(draw_cfg.get("keypoint_radius", 3))
            thick = int(draw_cfg.get("thickness", 2))
            # 標準的な COCO-17 の 0-based スケルトン定義
            # 参考: Ultralytics 等のデフォルト
            coco17_skeleton_0based = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)
            ]

            for det in detections:
                persons += 1
                skeleton_pairs = coco17_skeleton_0based if skel_on else None
                frame = draw_pose(
                    frame,
                    det["keypoints"],
                    skeleton=skeleton_pairs,
                    radius=kp_radius,
                    thickness=thick,
                    min_kpt_score=float(self.cfg.get("postprocess", {}).get("kpt_confidence", 0.20)),
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
    cfg_obj = load_settings(base_dir)

    disp_cfg = {
        "width": cfg_obj.display.width,
        "height": cfg_obj.display.height,
    }
    app = PoseViewerApp(
        display_w=int(disp_cfg.get("width", 960)),
        display_h=int(disp_cfg.get("height", 540)),
    )
    # 既存のControllerは辞書型設定を期待しているため、既存構造に合わせて辞書化
    cfg_dict = {
        "model": {
            "path": cfg_obj.model.path,
            "device": cfg_obj.model.device,
            "img_size": cfg_obj.model.img_size,
        },
        "input": {
            "source": cfg_obj.input.source,
            "max_frame": cfg_obj.input.max_frame,
            "capture": {
                "width": cfg_obj.input.capture.width,
                "height": cfg_obj.input.capture.height,
                "fps": cfg_obj.input.capture.fps,
            },
        },
        "performance": {"skip": cfg_obj.performance.skip},
        "postprocess": {
            "det_confidence": cfg_obj.postprocess.det_confidence,
            "kpt_confidence": cfg_obj.postprocess.kpt_confidence,
            "max_detections": cfg_obj.postprocess.max_detections,
            "nms_iou": cfg_obj.postprocess.nms_iou,
        },
        "draw": {
            "skeleton": cfg_obj.draw.skeleton,
            "keypoint_radius": cfg_obj.draw.keypoint_radius,
            "thickness": cfg_obj.draw.thickness,
        },
        "display": {"width": cfg_obj.display.width, "height": cfg_obj.display.height},
    }
    controller = CaptureController(app, cfg=cfg_dict, base_dir=base_dir)

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
    # 起動時に自動的に開始（Start 状態から開始）
    app.after(0, on_start)
    app.mainloop()


if __name__ == "__main__":
    main()
