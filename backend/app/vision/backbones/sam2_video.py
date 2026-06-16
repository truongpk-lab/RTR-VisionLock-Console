"""Optional streaming SAM2 video-predictor backbone.

Streams frames through SAM2's video predictor so SAM2's own cross-frame memory
attention carries the lock — the closest match to upstream SAMURAI. Needs a
streaming-capable SAM2 build + GPU, so it is OFF by default and any failure flips
``available`` off so :class:`ManagedTracker` falls back to OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional deployment dependency
    torch = None

try:  # Streaming video predictor is only present in newer SAM2 builds.
    from sam2.build_sam import build_sam2_video_predictor
except Exception:  # pragma: no cover - optional deployment dependency
    build_sam2_video_predictor = None

from ..utils import BBox, clamp_bbox
from .base import TrackResult, mask_to_bbox


class Sam2VideoBackbone:
    source = "sam2_video"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.available = False
        self.last_error = ""
        self.predictor = None
        self.state = None
        self._frame_idx = 0
        self.kind = "sam2_video"

        seg_cfg = config.get("models", {}).get("segmenter", {})
        if build_sam2_video_predictor is None or torch is None:
            self.last_error = "sam2 video predictor / torch not installed"
            return
        backend_root = Path(__file__).resolve().parents[3]
        checkpoint = backend_root / str(seg_cfg.get("checkpoint", ""))
        model_cfg = str(seg_cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml"))
        if not checkpoint.exists():
            self.last_error = f"SAM2 checkpoint not found: {checkpoint}"
            return
        try:
            device = str(config.get("runtime", {}).get("device", "cuda"))
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self.predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.predictor = None
            self.last_error = str(exc)

    def init(self, frame: np.ndarray, bbox: BBox) -> bool:
        if not self.available or self.predictor is None:
            return False
        try:
            height, width = frame.shape[:2]
            x, y, w, h = clamp_bbox(bbox, width, height)
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            # Streaming init: feed the first frame and the box prompt. The exact
            # call names can vary across SAM2 builds; guarded so failure -> fallback.
            self.state = self.predictor.init_state_from_frame(frame) if hasattr(
                self.predictor, "init_state_from_frame"
            ) else self.predictor.init_state(frame)
            self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=1, box=box)
            self._frame_idx = 0
            return True
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.last_error = str(exc)
            self.available = False
            return False

    def track(self, frame: np.ndarray) -> TrackResult:
        if not self.available or self.predictor is None or self.state is None:
            return TrackResult(False, None, 0.0, self.source)
        try:
            self._frame_idx += 1
            obj_ids, mask_logits = self.predictor.track(self.state, frame)
            mask = (mask_logits[0] > 0.0)
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            mask = np.asarray(mask).squeeze().astype(bool)
            bbox = mask_to_bbox(mask)
            if bbox is None:
                return TrackResult(False, None, 0.0, self.source)
            height, width = frame.shape[:2]
            return TrackResult(True, clamp_bbox(bbox, width, height), 1.0, self.source)
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.last_error = str(exc)
            self.available = False
            return TrackResult(False, None, 0.0, self.source)
