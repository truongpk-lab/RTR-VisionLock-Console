"""Adapter base for PyTracking / STARK-lineage deep trackers.

UETrack and EVPTrack (and OSTrack, SeqTrack, ODTrack ...) share the same
test-time interface inherited from the PyTracking framework::

    tracker.initialize(image_rgb, {"init_bbox": [x, y, w, h]})
    out = tracker.track(image_rgb)   # {"target_bbox": [x, y, w, h], "<score>": float}

The concrete repo lives outside this project (cloned on the Jetson). Rather than
vendor it, this adapter loads it dynamically from a configured ``repo_path`` via
``importlib`` and is fully guarded: any failure (no torch, repo not on disk,
checkpoint missing, API mismatch) flips ``available`` off so
:class:`ManagedTracker` falls back to OpenCV. Adding another tracker of this
family is one tiny subclass that fills in the default module/class names.

NOTE: the default module paths below follow the upstream repo layout. The exact
names can drift between forks, so every name is overridable from config
(``models.<config_key>.*``) and MUST be verified on-device during Jetson
bring-up — this code path cannot run on a dev box without torch + weights.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import torch
except Exception:  # pragma: no cover - optional deployment dependency
    torch = None

from ..utils import BBox, clamp_bbox
from .base import TrackResult

# Score keys deep trackers use, in order of preference.
_SCORE_KEYS = ("conf_score", "best_score", "max_score", "score", "confidence")


class StarkLineageBackbone:
    """Dynamically-loaded backbone for a PyTracking-style deep tracker."""

    # Subclasses set these; every value is also overridable from config.
    source = "torch_tracker"
    config_key = "tracker_torch"
    default_param_module = ""
    default_tracker_module = ""
    default_tracker_class = ""
    default_param_name = ""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.available = False
        self.last_error = ""
        self.tracker = None
        self.kind = self.source
        self._initialized = False

        block = config.get("models", {}).get(self.config_key, {})
        if not bool(block.get("enabled", True)):
            self.last_error = "disabled in config"
            return
        if torch is None or cv2 is None:
            self.last_error = "torch / cv2 not installed"
            return

        repo_path = str(block.get("repo_path", "")).strip()
        if not repo_path or not Path(repo_path).exists():
            self.last_error = f"repo_path not found: {repo_path or '(unset)'}"
            return
        checkpoint = self._resolve_path(block.get("checkpoint", ""))
        if checkpoint is None:
            self.last_error = "checkpoint not found"
            return

        try:
            self.tracker = self._build(block, repo_path, checkpoint)
            self.available = self.tracker is not None
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.tracker = None
            self.last_error = str(exc)

    # -- construction -------------------------------------------------------
    def _resolve_path(self, raw: Any) -> Path | None:
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path if path.exists() else None

    def _build(self, block: dict[str, Any], repo_path: str, checkpoint: Path):
        """Load the upstream tracker via importlib. Guarded by the caller."""
        for candidate in (repo_path, str(Path(repo_path) / "lib")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

        param_module = str(block.get("param_module", self.default_param_module))
        tracker_module = str(block.get("tracker_module", self.default_tracker_module))
        tracker_class = str(block.get("tracker_class", self.default_tracker_class))
        param_name = str(block.get("param_name", self.default_param_name))

        params = importlib.import_module(param_module).parameters(param_name)
        # Point the params at our checkpoint and device regardless of repo default.
        device = str(self.config.get("runtime", {}).get("device", "cuda"))
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        for attr in ("checkpoint", "net_path", "model"):
            if hasattr(params, attr):
                setattr(params, attr, str(checkpoint))
        if hasattr(params, "device"):
            params.device = device

        cls = getattr(importlib.import_module(tracker_module), tracker_class)
        tracker = cls(params, "video")
        self.kind = f"{self.source}:{param_name or tracker_class}"
        return tracker

    # -- runtime ------------------------------------------------------------
    @staticmethod
    def _to_rgb(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame

    def init(self, frame: np.ndarray, bbox: BBox) -> bool:
        if not self.available or self.tracker is None:
            return False
        try:
            x, y, w, h = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
            self.tracker.initialize(self._to_rgb(frame), {"init_bbox": [int(x), int(y), int(w), int(h)]})
            self._initialized = True
            return True
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.last_error = str(exc)
            self.available = False
            return False

    def track(self, frame: np.ndarray) -> TrackResult:
        if not self.available or self.tracker is None or not self._initialized:
            return TrackResult(False, None, 0.0, self.source)
        try:
            out = self.tracker.track(self._to_rgb(frame))
            box = out.get("target_bbox") if isinstance(out, dict) else out
            if box is None or len(box) < 4:
                return TrackResult(False, None, 0.0, self.source)
            bbox = clamp_bbox(
                (int(box[0]), int(box[1]), int(box[2]), int(box[3])), frame.shape[1], frame.shape[0]
            )
            score = 1.0
            if isinstance(out, dict):
                for key in _SCORE_KEYS:
                    if key in out and out[key] is not None:
                        score = float(out[key])
                        break
            score = max(0.0, min(1.0, score))
            ok = bbox[2] >= 4 and bbox[3] >= 4
            return TrackResult(ok, bbox if ok else None, score if ok else 0.0, self.source)
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.last_error = str(exc)
            self.available = False
            return TrackResult(False, None, 0.0, self.source)

    def to_dict(self) -> dict:
        return {"source": self.source, "kind": self.kind, "available": self.available, "last_error": self.last_error}


class UETrackBackbone(StarkLineageBackbone):
    """UETrack-B (CVPR 2026) — efficient unified SOT, ~60 FPS on Jetson AGX.

    Tier A normal/high-confidence tracker. See https://github.com/kangben258/UETrack.
    """

    source = "uetrack"
    config_key = "tracker_uetrack"
    default_param_module = "lib.test.parameter.uetrack"
    default_tracker_module = "lib.test.tracker.uetrack"
    default_tracker_class = "UETrack"
    default_param_name = "uetrack_b"


class EVPTrackBackbone(StarkLineageBackbone):
    """EVPTrack (AAAI 2024) — spatio-temporal explicit visual prompts.

    Tier B low-confidence local re-find tracker; strong under appearance change /
    deformation. See https://github.com/GXNU-ZhongLab/EVPTrack.
    """

    source = "evptrack"
    config_key = "tracker_evptrack"
    default_param_module = "lib.test.parameter.evptrack"
    default_tracker_module = "lib.test.tracker.evptrack"
    default_tracker_class = "EVPTrack"
    default_param_name = "baseline"
