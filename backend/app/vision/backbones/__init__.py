"""Swappable tracker backbones + the managed wrapper that drives one.

``BACKBONE_REGISTRY`` maps a config name to a backbone constructor. To add a new
tracker (e.g. a TensorRT engine), write one class with ``init``/``track`` and add
a single line here — the session, policy and UI are untouched.

``ManagedTracker`` owns exactly one backbone and the universal safety net: if the
requested backbone cannot load (no torch / no weights) or fails mid-stream, it
transparently drops to OpenCV so the lock is never left without an engine.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..utils import BBox, clamp_bbox
from .base import TrackResult, TrackerBackbone
from .opencv import OpenCvBackbone
from .sam2_video import Sam2VideoBackbone
from .torch_tracker import EVPTrackBackbone, UETrackBackbone

BACKBONE_REGISTRY: dict[str, Callable[[dict[str, Any]], TrackerBackbone]] = {
    "opencv": OpenCvBackbone,
    "sam2_video": Sam2VideoBackbone,
    "uetrack": UETrackBackbone,
    "evptrack": EVPTrackBackbone,
}


def build_backbone(name: str, config: dict[str, Any]) -> TrackerBackbone:
    """Construct a backbone by name, returning OpenCV for any unknown name."""
    factory = BACKBONE_REGISTRY.get(str(name).lower(), OpenCvBackbone)
    return factory(config)


def _resolve_default_name(config: dict[str, Any]) -> str:
    """Back-compat default used when no explicit backbone is requested."""
    samurai = config.get("samurai", {}) if isinstance(config, dict) else {}
    return "sam2_video" if bool(samurai.get("use_video_predictor", False)) else "opencv"


class ManagedTracker:
    """One backbone + OpenCV fallback behind a uniform tracker interface."""

    def __init__(self, config: dict[str, Any], backbone: str | None = None) -> None:
        self.config = config
        self.requested = backbone
        self.backbone = self._build()
        self.last_bbox: BBox | None = None

    def _build(self) -> TrackerBackbone:
        name = self.requested or _resolve_default_name(self.config)
        backbone = build_backbone(name, self.config)
        if getattr(backbone, "available", True):
            return backbone
        # Requested engine could not load -> universal OpenCV fallback.
        return OpenCvBackbone(self.config)

    @property
    def kind(self) -> str:
        return getattr(self.backbone, "kind", "none")

    @property
    def source(self) -> str:
        return getattr(self.backbone, "source", "opencv")

    def _is_fallback(self) -> bool:
        return self.source == "opencv"

    @property
    def is_opencv(self) -> bool:
        """The OpenCV backbone is running (no real confidence score; flat affinity).

        Used to decide that tracker confidence must be estimated from honest
        signals (identity + stability) rather than the backbone's own score.
        """
        return self._is_fallback()

    @property
    def is_fallback(self) -> bool:
        """A non-opencv backbone was requested but OpenCV is running -> silent fallback.

        Drives the operator-facing warning/badge. An explicit opencv request (or
        auto-resolve) is not a surprise, so it is not flagged.
        """
        requested = (self.requested or "").lower()
        return self.is_opencv and requested not in ("", "opencv")

    def init(self, frame: np.ndarray, bbox: BBox) -> bool:
        ok = self.backbone.init(frame, bbox)
        if not ok and not self._is_fallback():
            # Backbone refused (e.g. video/deep predictor failed) -> drop to OpenCV.
            self.backbone = OpenCvBackbone(self.config)
            ok = self.backbone.init(frame, bbox)
        if ok:
            self.last_bbox = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        return ok

    def track(self, frame: np.ndarray) -> TrackResult:
        result = self.backbone.track(frame)
        if result.ok and result.bbox is not None:
            self.last_bbox = result.bbox
        elif not result.ok and not self._is_fallback() and self.last_bbox is not None:
            # Mid-stream deep/video failure -> re-init OpenCV on the last good box.
            self.backbone = OpenCvBackbone(self.config)
            if self.backbone.init(frame, self.last_bbox):
                result = self.backbone.track(frame)
                if result.ok and result.bbox is not None:
                    self.last_bbox = result.bbox
        return result

    def to_dict(self) -> dict:
        info = {
            "source": self.source,
            "kind": self.kind,
            "requested": self.requested or "auto",
            "fallback": self.is_fallback,
        }
        last_error = getattr(self.backbone, "last_error", "")
        if last_error:
            info["last_error"] = last_error
        return info


__all__ = [
    "BACKBONE_REGISTRY",
    "ManagedTracker",
    "TrackResult",
    "build_backbone",
]
