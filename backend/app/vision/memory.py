from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .utils import BBox, clamp_bbox


class MemoryBank:
    def __init__(self, config: dict[str, Any]) -> None:
        memory_cfg = config.get("memory", {})
        self.ram_slots = int(memory_cfg.get("ram_slots", 8))
        self.drm_slots = int(memory_cfg.get("drm_slots", 8))
        self.feature_dim = int(memory_cfg.get("feature_dim", 1024))
        self.base_id = f"{memory_cfg.get('base_id_prefix', 'TGT')}-8842-A"
        self.ram: deque[np.ndarray] = deque(maxlen=self.ram_slots)
        self.drm: deque[np.ndarray] = deque(maxlen=self.drm_slots)

    def extract(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        if cv2 is None or frame is None:
            return None
        height, width = frame.shape[:2]
        x, y, w, h = clamp_bbox(bbox, width, height)
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype("float32")

    def initialize(self, frame: np.ndarray, bbox: BBox) -> bool:
        feature = self.extract(frame, bbox)
        if feature is None:
            return False
        self.ram.clear()
        self.ram.append(feature)
        return True

    def load_samples(self, samples: list[np.ndarray]) -> bool:
        """Seed RAM/DRM from features collected during LEARNING_TARGET."""
        clean = [s for s in samples if s is not None]
        if not clean:
            return False
        self.ram.clear()
        self.drm.clear()
        # Newest samples are the most representative of the locked appearance.
        for feature in clean[-self.ram_slots :]:
            self.ram.append(feature)
        # Spread a few earlier samples into DRM as a longer-term reference.
        for feature in clean[:: max(1, len(clean) // self.drm_slots or 1)][: self.drm_slots]:
            self.drm.append(feature)
        return True

    def update_ram(self, frame: np.ndarray, bbox: BBox) -> None:
        feature = self.extract(frame, bbox)
        if feature is not None:
            self.ram.append(feature)

    def update_drm(self, frame: np.ndarray, bbox: BBox) -> None:
        feature = self.extract(frame, bbox)
        if feature is not None:
            self.drm.append(feature)

    def similarity(self, frame: np.ndarray, bbox: BBox) -> float:
        if cv2 is None or not self.ram:
            return 0.0
        feature = self.extract(frame, bbox)
        if feature is None:
            return 0.0
        scores = [float(cv2.compareHist(feature, item, cv2.HISTCMP_CORREL)) for item in self.ram]
        score = max(scores) if scores else 0.0
        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict:
        return {
            "base_id": self.base_id,
            "feature_dim": self.feature_dim,
            "ram_slots": len(self.ram),
            "ram_capacity": self.ram_slots,
            "drm_slots": len(self.drm),
            "drm_capacity": self.drm_slots,
            "ram_enabled": True,
            "drm_enabled": True,
        }
