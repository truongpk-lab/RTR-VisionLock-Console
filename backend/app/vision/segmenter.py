"""Promptable click-to-segment for target selection.

A single click should yield ONE coherent object box instead of the noisy
contour fragments the edge-based proposal produced. This implements the
"Segment Anything" point-prompt paradigm (Kirillov et al., 2023) using a
lightweight ONNX SAM variant (MobileSAM / EfficientViT-SAM) on CPU, and falls
back to OpenCV GrabCut around the click when no model is present, so
click-to-select always works without a model download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

from .utils import BBox, clamp_bbox

# SAM normalises pixels with these ImageNet statistics before the encoder.
_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)


class PromptableSegmenter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        seg_cfg = config.get("models", {}).get("segmenter", {})
        self.input_size = int(seg_cfg.get("input_size", 1024))
        self.encoder = None
        self.decoder = None
        self.backend = "grabcut"

        if seg_cfg.get("enabled") is True and ort is not None:
            backend_root = Path(__file__).resolve().parents[2]
            enc = backend_root / str(seg_cfg.get("encoder_path", ""))
            dec = backend_root / str(seg_cfg.get("decoder_path", ""))
            if enc.exists() and dec.exists():
                try:
                    providers = ["CPUExecutionProvider"]
                    self.encoder = ort.InferenceSession(str(enc), providers=providers)
                    self.decoder = ort.InferenceSession(str(dec), providers=providers)
                    self.backend = "sam_onnx"
                except Exception:
                    self.encoder = None
                    self.decoder = None
                    self.backend = "grabcut"

    @property
    def ready_model(self) -> bool:
        return self.backend == "sam_onnx"

    def segment_point(self, frame: np.ndarray, point: tuple[int, int]) -> BBox | None:
        """Return one object bbox at the clicked point, or None."""
        if cv2 is None or frame is None:
            return None
        height, width = frame.shape[:2]
        px = max(0, min(int(point[0]), width - 1))
        py = max(0, min(int(point[1]), height - 1))
        if self.backend == "sam_onnx":
            bbox = self._segment_sam(frame, (px, py))
            if bbox is not None:
                return clamp_bbox(bbox, width, height)
        return self._segment_grabcut(frame, (px, py))

    # --- SAM ONNX path -----------------------------------------------------
    def _segment_sam(self, frame: np.ndarray, point: tuple[int, int]) -> BBox | None:
        try:
            height, width = frame.shape[:2]
            scale = self.input_size / float(max(height, width))
            new_w, new_h = int(round(width * scale)), int(round(height * scale))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            norm = (resized.astype(np.float32) - _PIXEL_MEAN) / _PIXEL_STD
            padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
            padded[:new_h, :new_w] = norm
            blob = np.transpose(padded, (2, 0, 1))[None]  # [1,3,S,S]

            enc_input = self.encoder.get_inputs()[0].name
            embedding = self.encoder.run(None, {enc_input: blob})[0]

            # Point in resized-image coordinates, plus the SAM padding point.
            coords = np.array([[point[0] * scale, point[1] * scale], [0.0, 0.0]], dtype=np.float32)[None]
            labels = np.array([1, -1], dtype=np.float32)[None]
            dec_inputs = {
                "image_embeddings": embedding.astype(np.float32),
                "point_coords": coords,
                "point_labels": labels,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": np.array([height, width], dtype=np.float32),
            }
            # Only feed inputs the decoder actually declares (export variants differ).
            declared = {i.name for i in self.decoder.get_inputs()}
            dec_inputs = {k: v for k, v in dec_inputs.items() if k in declared}
            outputs = self.decoder.run(None, dec_inputs)
            masks = outputs[0]  # [1, k, H, W] logits
            iou = outputs[1] if len(outputs) > 1 else None
            best = int(np.argmax(iou[0])) if iou is not None else 0
            mask = masks[0, best] > 0.0
            return self._mask_to_bbox(mask)
        except Exception:
            return None

    # --- GrabCut fallback --------------------------------------------------
    def _segment_grabcut(self, frame: np.ndarray, point: tuple[int, int]) -> BBox | None:
        height, width = frame.shape[:2]
        # Work on a downscaled copy: GrabCut is heavy at full resolution.
        scale = 512.0 / max(height, width)
        scale = min(1.0, scale)
        work = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
        wh, ww = work.shape[:2]
        cx, cy = int(point[0] * scale), int(point[1] * scale)

        # Seed rectangle centred on the click (~40% of the smaller side).
        half = max(12, int(0.20 * min(wh, ww)))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(ww - 1, cx + half), min(wh - 1, cy + half)
        rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

        mask = np.zeros((wh, ww), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return self._fallback_box(point, width, height)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        bbox = self._mask_to_bbox(fg.astype(bool))
        if bbox is None:
            return self._fallback_box(point, width, height)
        inv = 1.0 / scale
        x, y, w, h = bbox
        scaled = (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
        return clamp_bbox(scaled, width, height)

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> BBox | None:
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w < 6 or h < 6:
            return None
        return (x0, y0, w, h)

    @staticmethod
    def _fallback_box(point: tuple[int, int], width: int, height: int) -> BBox:
        # Last resort so a click always selects something near the cursor.
        side = max(24, int(0.12 * min(width, height)))
        x = point[0] - side // 2
        y = point[1] - side // 2
        return clamp_bbox((x, y, side, side), width, height)

    def to_dict(self) -> dict:
        return {"backend": self.backend, "model_ready": self.ready_model}
