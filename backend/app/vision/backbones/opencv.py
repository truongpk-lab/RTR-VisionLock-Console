"""OpenCV motion backbone (CSRT/KCF/MOSSE).

Dependency-free and always available, so it is the universal fallback for every
deep backbone that fails to load. The SAMURAI motion-aware mask selection and
memory admission still run on top of it via the session's periodic SAM2 refine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..tracker import OpenCVTracker
from ..utils import BBox
from .base import TrackResult


class OpenCvBackbone:
    source = "opencv"
    available = True

    def __init__(self, config: dict[str, Any]) -> None:
        self._tracker = OpenCVTracker(config)

    @property
    def kind(self) -> str:
        return self._tracker.kind

    def init(self, frame: np.ndarray, bbox: BBox) -> bool:
        return self._tracker.init(frame, bbox)

    def track(self, frame: np.ndarray) -> TrackResult:
        ok, bbox = self._tracker.update(frame)
        return TrackResult(ok=ok, bbox=bbox, affinity=1.0 if ok else 0.0, source=self.source)
