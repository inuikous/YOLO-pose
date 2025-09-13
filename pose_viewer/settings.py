"""アプリ設定の読み込み/検証モジュール（最小実装）。

YAMLを読み込み、型付き設定として提供する。欠損項目には安全なデフォルトを適用する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class ModelSettings:
    path: str
    device: str = "CPU"
    img_size: int = 640


@dataclass
class CaptureSettings:
    width: int = 0
    height: int = 0
    fps: int = 0


@dataclass
class InputSettings:
    source: Any = 0
    max_frame: int = 0
    capture: CaptureSettings = field(default_factory=CaptureSettings)


@dataclass
class PerformanceSettings:
    skip: int = 0


@dataclass
class PostprocessSettings:
    det_confidence: float = 0.25
    kpt_confidence: float = 0.20
    max_detections: int = 10
    nms_iou: float = 0.45


@dataclass
class DrawSettings:
    skeleton: bool = True
    keypoint_radius: int = 3
    thickness: int = 2


@dataclass
class DisplaySettings:
    width: int = 1200
    height: int = 900


@dataclass
class AppSettings:
    model: ModelSettings
    input: InputSettings
    performance: PerformanceSettings
    postprocess: PostprocessSettings
    draw: DrawSettings
    display: DisplaySettings


def _as_int(d: Dict[str, Any], key: str, default: int) -> int:
    try:
        v = d.get(key, default)
        return int(v)
    except Exception:
        return default


def _as_float(d: Dict[str, Any], key: str, default: float) -> float:
    try:
        v = d.get(key, default)
        return float(v)
    except Exception:
        return default


def _capture_from_dict(d: Dict[str, Any]) -> CaptureSettings:
    return CaptureSettings(
        width=_as_int(d, "width", 0),
        height=_as_int(d, "height", 0),
        fps=_as_int(d, "fps", 0),
    )


def load_settings(base_dir: Path, filename: str = "config.yaml") -> AppSettings:
    cfg_path = base_dir / filename
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # model
    model_raw = raw.get("model", {}) or {}
    model = ModelSettings(
        path=str(model_raw.get("path", "models/test.xml")),
        device=str(model_raw.get("device", "CPU")),
        img_size=_as_int(model_raw, "img_size", 640),
    )

    # input
    input_raw = raw.get("input", {}) or {}
    capture = _capture_from_dict(input_raw.get("capture", {}) or {})
    input_cfg = InputSettings(
        source=input_raw.get("source", 0),
        max_frame=_as_int(input_raw, "max_frame", 0),
        capture=capture,
    )

    # performance
    perf_raw = raw.get("performance", {}) or {}
    performance = PerformanceSettings(skip=_as_int(perf_raw, "skip", 0))

    # postprocess
    pp_raw = raw.get("postprocess", {}) or {}
    postprocess = PostprocessSettings(
        det_confidence=_as_float(pp_raw, "det_confidence", 0.25),
        kpt_confidence=_as_float(pp_raw, "kpt_confidence", 0.20),
        max_detections=_as_int(pp_raw, "max_detections", 10),
        nms_iou=_as_float(pp_raw, "nms_iou", 0.45),
    )

    # draw
    draw_raw = raw.get("draw", {}) or {}
    draw = DrawSettings(
        skeleton=bool(draw_raw.get("skeleton", True)),
        keypoint_radius=_as_int(draw_raw, "keypoint_radius", 3),
        thickness=_as_int(draw_raw, "thickness", 2),
    )

    # display
    disp_raw = raw.get("display", {}) or {}
    display = DisplaySettings(
        width=_as_int(disp_raw, "width", 1200),
        height=_as_int(disp_raw, "height", 900),
    )

    return AppSettings(
        model=model,
        input=input_cfg,
        performance=performance,
        postprocess=postprocess,
        draw=draw,
        display=display,
    )
