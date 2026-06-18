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
from .torch_tracker import UETrackBackbone
from .uetrack_onnx import UETrackOnnxBackbone

BACKBONE_REGISTRY: dict[str, Callable[[dict[str, Any]], TrackerBackbone]] = {
    "opencv": OpenCvBackbone,
    "sam2_video": Sam2VideoBackbone,
    "uetrack": UETrackBackbone,
    "uetrack_onnx": UETrackOnnxBackbone,
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
        self._warmed = False

    def _build(self) -> TrackerBackbone:
        name = self.requested or _resolve_default_name(self.config)
        backbone = build_backbone(name, self.config)
        if getattr(backbone, "available", True):
            return backbone
        # Requested engine could not load -> universal OpenCV fallback.
        return OpenCvBackbone(self.config)

    def _wants_deep(self) -> bool:
        """A non-opencv backbone was explicitly requested."""
        return (self.requested or "").lower() not in ("", "opencv")

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

    def reinit(self, frame: np.ndarray, bbox: BBox, *, try_recover_deep: bool = True) -> bool:
        """Re-seed the EXISTING backbone on a new bbox without rebuilding the network.

        This is the hot path for every (re)lock: a deep tracker's ``initialize`` is
        cheap (~ms), while rebuilding it reloads weights and a CUDA/ONNX session
        (hundreds of ms). So we keep one backbone alive and only re-seed it.

        One exception: if the backbone silently dropped to OpenCV mid-stream (a
        transient deep failure) and a deep engine was requested, we try ONE rebuild
        here so the deep tracker can recover. If that rebuild also fails we keep the
        OpenCV fallback, so behaviour never regresses.
        """
        recover = try_recover_deep and bool(
            self.config.get("tracking", {}).get("recover_deep_on_relock", True)
        )
        if recover and self._is_fallback() and self._wants_deep():
            rebuilt = self._build()
            if getattr(rebuilt, "available", True) and rebuilt.init(frame, bbox):
                self.backbone = rebuilt
                self.last_bbox = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
                self._warmed = True  # the recovery init already paid the first-call cost
                return True
            # Rebuild unavailable / failed -> fall through to re-seed the OpenCV fallback.
        return self.init(frame, bbox)

    def warmup(self, frame_shape: tuple[int, int]) -> None:
        """Run ONE dummy init+track to build the engine / init CUDA before the user locks.

        Idempotent and never raises: a deep backbone's first inference builds the
        TensorRT engine and initialises the CUDA context, which is what otherwise
        stalls the first real lock. OpenCV / torch-less boxes are a cheap no-op.
        """
        if self._warmed:
            return
        self._warmed = True
        if self._is_fallback():  # OpenCV: nothing heavy to warm
            return
        try:
            h, w = int(frame_shape[0]), int(frame_shape[1])
            if h < 16 or w < 16:
                return
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            bbox = (w // 2 - 20, h // 2 - 20, 40, 40)
            if self.backbone.init(dummy, bbox):
                self.backbone.track(dummy)  # force a real forward -> builds the engine
        except Exception:  # pragma: no cover - warmup must never break start
            pass
        finally:
            self.last_bbox = None  # discard the dummy seed so the real lock re-seeds clean

    def close(self) -> None:
        """Drop the backbone and free GPU memory (best-effort) on a hard stop."""
        self.backbone = None
        self.last_bbox = None
        self._warmed = False
        try:  # pragma: no cover - depends on deployment runtime
            import torch

            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
