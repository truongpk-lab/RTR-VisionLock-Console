"""Streaming SAM2 backbone (SAMURAI-style real-time tracking).

Streams frames through SAM2's per-frame memory attention so the lock is carried
by SAM2's own cross-frame memory — the closest drop-in to upstream SAMURAI for a
live camera. Uses the real-time *camera predictor* API
(``build_sam2_camera_predictor`` / ``load_first_frame`` / ``add_new_prompt`` /
``track``) from the streaming SAM2 build; falls back to the offline video
predictor API if only that is present. Any failure (no package / no checkpoint /
no GPU op) flips ``available`` off so :class:`ManagedTracker` drops to OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional deployment dependency
    torch = None

try:  # Real-time streaming predictor (segment-anything-2-real-time / SAMURAI).
    from sam2.build_sam import build_sam2_camera_predictor
except Exception:  # pragma: no cover - optional deployment dependency
    build_sam2_camera_predictor = None

try:  # Offline video predictor (vanilla Meta SAM2) — used only as a fallback.
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
        self._streaming = False  # True -> camera predictor (per-frame track(img))
        self._frame_idx = 0
        self.kind = "sam2_video"

        seg_cfg = config.get("models", {}).get("segmenter", {})
        if torch is None or (build_sam2_camera_predictor is None and build_sam2_video_predictor is None):
            self.last_error = "sam2 streaming predictor / torch not installed"
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
            # Lower the SAM2 input size for real-time on modest GPUs: 1024 (default)
            # is ~4-5 FPS on a laptop 4060; 512 is ~25-30 FPS with little accuracy
            # loss for a single box-prompted target. 0 keeps the model default.
            stream_size = int(seg_cfg.get("stream_image_size", 512))
            overrides = [f"++model.image_size={stream_size}"] if stream_size else []
            if build_sam2_camera_predictor is not None:
                self.predictor = build_sam2_camera_predictor(
                    model_cfg, str(checkpoint), device=device, hydra_overrides_extra=overrides
                )
                self._streaming = True
                self.kind = f"sam2_camera@{stream_size}" if stream_size else "sam2_camera"
            else:
                self.predictor = build_sam2_video_predictor(
                    model_cfg, str(checkpoint), device=device, hydra_overrides_extra=overrides
                )
                self.kind = f"sam2_video@{stream_size}" if stream_size else "sam2_video"
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
            self._frame_idx = 0
            if self._streaming:
                # Real-time camera predictor: feed the first frame, then the box prompt.
                self.predictor.load_first_frame(frame)
                self.predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=box)
                self.state = True
                return True
            # Offline video predictor fallback.
            self.state = self.predictor.init_state_from_frame(frame) if hasattr(
                self.predictor, "init_state_from_frame"
            ) else self.predictor.init_state(frame)
            self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=1, box=box)
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
            if self._streaming:
                _obj_ids, mask_logits = self.predictor.track(frame)
            else:
                _obj_ids, mask_logits = self.predictor.track(self.state, frame)
            mask = mask_logits[0] > 0.0
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
