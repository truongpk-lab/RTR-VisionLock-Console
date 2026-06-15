from __future__ import annotations

import math
from typing import Any

from app.core.metrics import clamp

from .utils import BBox, bbox_center


class SimpleKalmanGate:
    def __init__(self, config: dict[str, Any]) -> None:
        self.max_error = float(config.get("thresholds", {}).get("kalman_max_error", 80))
        self.bbox: BBox | None = None
        self.velocity = (0.0, 0.0)

    def reset(self, bbox: BBox | None = None) -> None:
        self.bbox = bbox
        self.velocity = (0.0, 0.0)

    def predict(self) -> BBox | None:
        if self.bbox is None:
            return None
        x, y, w, h = self.bbox
        vx, vy = self.velocity
        return (int(x + vx), int(y + vy), w, h)

    def update(self, bbox: BBox) -> tuple[float, float]:
        predicted = self.predict() or bbox
        pcx, pcy = bbox_center(predicted)
        cx, cy = bbox_center(bbox)
        error = math.hypot(cx - pcx, cy - pcy)
        if self.bbox is not None:
            old_cx, old_cy = bbox_center(self.bbox)
            self.velocity = (0.7 * self.velocity[0] + 0.3 * (cx - old_cx), 0.7 * self.velocity[1] + 0.3 * (cy - old_cy))
        self.bbox = bbox
        consistency = clamp(1.0 - error / max(1.0, self.max_error))
        return error, consistency

    def score_candidate(self, bbox: BBox) -> float:
        predicted = self.predict()
        if predicted is None:
            return 1.0
        pcx, pcy = bbox_center(predicted)
        cx, cy = bbox_center(bbox)
        error = math.hypot(cx - pcx, cy - pcy)
        return clamp(1.0 - error / max(1.0, self.max_error * 2.0))
