"""Promptable target refinement for YOLO-first selection.

The default flow is detector first: YOLO proposes candidate boxes, then SAM2
receives the chosen box plus a positive click and nearby distractor prompts.
Manual point segmentation remains as a fallback when no detector/SAM2 runtime
is available.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
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

from .motion import MaskCandidate, MotionWeights, select_motion_aware_mask
from .utils import BBox, bbox_center, bbox_iou, clamp_bbox

try:
    import torch
except Exception:  # pragma: no cover - optional deployment dependency
    torch = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:  # pragma: no cover - optional deployment dependency
    build_sam2 = None
    SAM2ImagePredictor = None

# SAM normalises pixels with these ImageNet statistics before the encoder.
_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class SegmentResult:
    bbox: BBox
    quality: float
    backend: str
    refined: bool = True
    polygon: list[list[int]] | None = None
    # Selected boolean foreground mask (full-frame, ``bool``). Only SAM2 sets it;
    # used to exclude shadow/background when building the appearance template.
    mask: np.ndarray | None = None


class PromptableSegmenter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        seg_cfg = config.get("models", {}).get("segmenter", {})
        self.input_size = int(seg_cfg.get("input_size", 1024))
        self.refine_interval = int(config.get("runtime", {}).get("sam_refine_interval", 8))
        self.video_memory_window = int(config.get("sam2", {}).get("video_memory_window", 16))
        self.motion_weights = MotionWeights.from_config(config)
        self.encoder = None
        self.decoder = None
        self.sam2_predictor = None
        self.checkpoint = str(seg_cfg.get("checkpoint", ""))
        self.last_error = ""
        self.backend = "grabcut"

        if seg_cfg.get("enabled") is True:
            kind = str(seg_cfg.get("type", "sam2")).lower()
            if kind == "sam2" and self._init_sam2(seg_cfg):
                return
            self._init_sam_onnx(seg_cfg)

    @property
    def ready_model(self) -> bool:
        return self.backend in {"sam2", "sam_onnx"}

    def _init_sam2(self, seg_cfg: dict[str, Any]) -> bool:
        if build_sam2 is None or SAM2ImagePredictor is None or torch is None:
            self.last_error = "sam2/torch package is not installed"
            return False
        backend_root = Path(__file__).resolve().parents[2]
        checkpoint = backend_root / str(seg_cfg.get("checkpoint", ""))
        model_cfg = str(seg_cfg.get("model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml"))
        if not checkpoint.exists():
            self.last_error = f"SAM2 checkpoint not found: {checkpoint}"
            return False
        try:
            device = str(self.config.get("runtime", {}).get("device", "cuda"))
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            model = build_sam2(model_cfg, str(checkpoint), device=device)
            self.sam2_predictor = SAM2ImagePredictor(model)
            self.backend = "sam2"
            self.checkpoint = str(checkpoint)
            return True
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.sam2_predictor = None
            self.last_error = str(exc)
            return False

    def _init_sam_onnx(self, seg_cfg: dict[str, Any]) -> bool:
        if ort is None:
            return False
        backend_root = Path(__file__).resolve().parents[2]
        enc = backend_root / str(seg_cfg.get("encoder_path", ""))
        dec = backend_root / str(seg_cfg.get("decoder_path", ""))
        if not enc.exists() or not dec.exists():
            return False
        try:
            providers = ["CPUExecutionProvider"]
            self.encoder = ort.InferenceSession(str(enc), providers=providers)
            self.decoder = ort.InferenceSession(str(dec), providers=providers)
            self.backend = "sam_onnx"
            return True
        except Exception as exc:
            self.encoder = None
            self.decoder = None
            self.backend = "grabcut"
            self.last_error = str(exc)
            return False

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

    def refine_box(
        self,
        frame: np.ndarray,
        bbox: BBox,
        positive_point: tuple[int, int] | None = None,
        negative_boxes: list[BBox] | None = None,
        motion_bbox: BBox | None = None,
    ) -> SegmentResult | None:
        """Refine a detector bbox with SAM2 box/click prompts, or fallback.

        Negative boxes are converted to negative point prompts for SAM2. The
        fallback path uses them as a veto against masks that grow into nearby
        same-class distractors. ``motion_bbox`` (the Kalman prediction) is the
        SAMURAI motion prior used to disambiguate SAM2's multimask output.
        """
        if cv2 is None or frame is None:
            return None
        height, width = frame.shape[:2]
        bbox = clamp_bbox(bbox, width, height)
        if self.backend == "sam2" and self.sam2_predictor is not None:
            result = self._refine_sam2(frame, bbox, positive_point, negative_boxes or [], motion_bbox)
            if result is not None:
                return result
        refined = self._refine_grabcut_box(frame, bbox)
        if refined is None:
            return SegmentResult(bbox=bbox, quality=0.0, backend=self.backend, refined=False)
        if self._overlaps_distractor(refined, negative_boxes or []):
            return SegmentResult(bbox=bbox, quality=0.0, backend=self.backend, refined=False)
        quality = bbox_iou(bbox, refined)
        return SegmentResult(bbox=clamp_bbox(refined, width, height), quality=quality, backend=self.backend, refined=refined != bbox)

    def _refine_sam2(
        self,
        frame: np.ndarray,
        bbox: BBox,
        positive_point: tuple[int, int] | None,
        negative_boxes: list[BBox],
        motion_bbox: BBox | None = None,
    ) -> SegmentResult | None:
        try:
            height, width = frame.shape[:2]
            x, y, w, h = clamp_bbox(bbox, width, height)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam2_predictor.set_image(rgb)
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            point_coords, point_labels = self._prompt_points((x, y, w, h), positive_point, negative_boxes)
            masks, scores, _ = self.sam2_predictor.predict(
                box=box,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                return None
            # SAMURAI mask selection: prefer the mask agreeing with the motion
            # prediction (Kalman), falling back to the prompt box before motion exists.
            candidates: list[MaskCandidate] = []
            mask_by_bbox: dict[tuple, np.ndarray] = {}
            for idx, mask in enumerate(masks):
                mask_bool = mask.astype(bool)
                candidate_bbox = self._mask_to_bbox(mask_bool)
                if candidate_bbox is None:
                    continue
                affinity = float(scores[idx]) if scores is not None and len(scores) > idx else bbox_iou(bbox, candidate_bbox)
                candidates.append(MaskCandidate(bbox=candidate_bbox, affinity=affinity))
                mask_by_bbox.setdefault(tuple(candidate_bbox), mask_bool)
            selection = select_motion_aware_mask(
                candidates,
                predicted_bbox=motion_bbox or bbox,
                negatives=negative_boxes,
                weights=self.motion_weights,
            )
            if selection is None:
                return None
            refined = selection.bbox
            if self._overlaps_distractor(refined, negative_boxes):
                return SegmentResult(bbox=clamp_bbox(bbox, width, height), quality=0.0, backend="sam2", refined=False)
            quality = selection.affinity if scores is not None and len(scores) else bbox_iou(bbox, refined)
            mask_bool = mask_by_bbox.get(tuple(refined))
            polygon = self._mask_to_polygon(mask_bool)
            return SegmentResult(
                bbox=clamp_bbox(refined, width, height),
                quality=max(0.0, min(1.0, quality)),
                backend="sam2",
                polygon=polygon,
                mask=mask_bool,
            )
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.last_error = str(exc)
            return None

    def _prompt_points(
        self,
        bbox: BBox,
        positive_point: tuple[int, int] | None,
        negative_boxes: list[BBox],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        points: list[tuple[float, float]] = []
        labels: list[int] = []
        if positive_point is None:
            positive_point = tuple(int(v) for v in bbox_center(bbox))
        points.append((float(positive_point[0]), float(positive_point[1])))
        labels.append(1)
        for negative in negative_boxes[:8]:
            points.append(bbox_center(negative))
            labels.append(0)
        if not points:
            return None, None
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

    @staticmethod
    def _overlaps_distractor(bbox: BBox, negative_boxes: list[BBox]) -> bool:
        return any(bbox_iou(bbox, negative) >= 0.30 for negative in negative_boxes)

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

    def _refine_grabcut_box(self, frame: np.ndarray, bbox: BBox) -> BBox | None:
        height, width = frame.shape[:2]
        x, y, w, h = clamp_bbox(bbox, width, height)
        if w < 6 or h < 6:
            return bbox
        scale = 512.0 / max(height, width)
        scale = min(1.0, scale)
        work = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()
        rect = (
            max(0, int(x * scale)),
            max(0, int(y * scale)),
            max(1, int(w * scale)),
            max(1, int(h * scale)),
        )
        mask = np.zeros(work.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return bbox
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(bool)
        refined = self._mask_to_bbox(fg)
        if refined is None:
            return bbox
        inv = 1.0 / scale
        rx, ry, rw, rh = refined
        return clamp_bbox((int(rx * inv), int(ry * inv), int(rw * inv), int(rh * inv)), width, height)

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray | None, max_points: int = 60) -> list[list[int]] | None:
        """Largest-contour outline of a boolean mask, simplified for the UI overlay."""
        if mask is None or cv2 is None:
            return None
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            return None
        epsilon = 0.01 * cv2.arcLength(contour, True)
        pts = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(pts) > max_points:
            step = int(np.ceil(len(pts) / max_points))
            pts = pts[::step]
        return [[int(x), int(y)] for x, y in pts]

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
        return {
            "backend": self.backend,
            "model_ready": self.ready_model,
            "checkpoint": self.checkpoint,
            "refine_interval": self.refine_interval,
            "video_memory_window": self.video_memory_window,
            "last_error": self.last_error,
        }
