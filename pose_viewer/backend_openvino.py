"""OpenVINO backend (dummy implementation for initial GUI integration).

Later this will:
 1. Load OpenVINO compiled model from XML (and BIN) using openvino.runtime.Core
 2. Preprocess frame (resize, BGR->RGB, float32, normalize)
 3. Run inference and decode YOLO pose outputs -> list of detections

Current dummy behaviour:
  - Returns a single pseudo person with synthetic moving keypoints (for GUI test).
"""

from __future__ import annotations

import math
import time
from typing import List, Dict, Any

import numpy as np


class OpenVinoPoseBackend:
    def __init__(self, model_xml_path: str, device: str = "CPU", img_size: int = 640):
        self.model_xml_path = model_xml_path
        self.device = device
        self.img_size = img_size
        self._start_time = time.time()
        # Placeholder for future: load OpenVINO model
        # from openvino.runtime import Core
        # core = Core()
        # model = core.read_model(model=model_xml_path)
        # self.compiled = core.compile_model(model=model, device_name=device)
        # self.input_layer = self.compiled.input(0)
        # self.output_layer = self.compiled.output(0)

    def infer(self, bgr_image) -> List[Dict[str, Any]]:
        """Return dummy detections list.

        Each detection dict fields:
          bbox: (x1,y1,x2,y2)
          keypoints: list[(x,y,score)] (COCO17 style length=17) synthetic
          score: float
          class_id: int
        """
        h, w = bgr_image.shape[:2]
        elapsed = time.time() - self._start_time

        # Synthetic oscillation for keypoints
        cx = w * 0.5 + math.sin(elapsed) * (w * 0.05)
        cy = h * 0.5 + math.cos(elapsed * 0.7) * (h * 0.05)
        bw = w * 0.35
        bh = h * 0.55
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # Generate 17 keypoints distributed inside bbox
        kpts = []
        rng = np.random.default_rng(int(elapsed * 10) % 100000)
        for i in range(17):
            kx = rng.uniform(x1 + 0.1 * bw, x2 - 0.1 * bw)
            ky = rng.uniform(y1 + 0.05 * bh, y2 - 0.05 * bh)
            score = rng.uniform(0.7, 0.99)
            kpts.append((float(kx), float(ky), float(score)))

        return [
            {
                "bbox": (x1, y1, x2, y2),
                "keypoints": kpts,
                "score": 0.9,
                "class_id": 0,
            }
        ]

    def close(self):  # future resource cleanup
        pass
