"""Shared types for swappable tracker backbones.

A *backbone* is one motion/appearance engine that can carry the locked target:
OpenCV CSRT, a streaming SAM2 video predictor, or a deep tracker (UETrack-B,
EVPTrack). They all speak the same tiny contract so the rest of the app never
branches on which engine is live:

    init(frame, bbox) -> bool          # seed the tracker on a box
    track(frame)      -> TrackResult   # advance one frame

``available`` lets a backbone declare it could not load (missing torch/weights)
so :class:`ManagedTracker` can transparently fall back to OpenCV. Adding a new
backbone means writing one class + registering it in ``backbones/__init__.py``;
nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..utils import BBox


@dataclass
class TrackResult:
    ok: bool
    bbox: BBox | None
    # 0..1 confidence reported by the backbone itself (OpenCV reports 1/0; deep
    # trackers report their real score). The session blends this with its Kalman
    # jitter / identity scores when grading the frame.
    affinity: float
    source: str


@runtime_checkable
class TrackerBackbone(Protocol):
    """Structural contract every backbone implements."""

    source: str
    available: bool

    @property
    def kind(self) -> str: ...

    def init(self, frame: np.ndarray, bbox: BBox) -> bool: ...

    def track(self, frame: np.ndarray) -> TrackResult: ...


def mask_to_bbox(mask: np.ndarray) -> BBox | None:
    """Tightest integer bbox around a boolean mask, or ``None`` if degenerate."""
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    if w < 4 or h < 4:
        return None
    return (x0, y0, w, h)
