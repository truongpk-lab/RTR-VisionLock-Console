"""Pluggable appearance/identity encoders.

The memory bank scores how much a crop looks like the locked target, and that
score is only as good as the feature it compares. Making the encoder a small
swappable component keeps that quality improvable without touching the memory
bank: add a subclass and a branch in ``build_encoder``.

* ``hsv_shape``         — original lightweight global HSV histogram + shape.
                          Kept for backward compatibility / lowest cost.
* ``hsv_block_texture`` — default. Spatial HSV blocks + gradient-orientation
                          texture, far better at telling apart two objects of the
                          same class/colour, still CPU-only and Jetson-cheap.

Every encoder returns an L2-normalised ``float32`` vector (or ``None`` when the
crop is unusable), so the cosine similarity in ``MemoryBank`` is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:  # optional: deep ReID encoder runs on onnxruntime (no torch needed)
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from .utils import BBox, clamp_bbox

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _suppress_background(crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    """Mean-fill background pixels for a dense (deep) encoder.

    A hard black fill is off-distribution for an ImageNet/OSNet-trained net and
    creates a synthetic mask boundary; filling with the foreground's mean colour
    and dilating the mask a few pixels keeps a thin silhouette margin while
    removing the shadow/road context that would bias the embedding.
    """
    fg = crop_mask > 0
    if not bool(fg.any()):
        return crop
    kernel = np.ones((5, 5), np.uint8)
    keep = cv2.dilate(crop_mask, kernel, iterations=1) > 0
    mean = crop[fg].reshape(-1, crop.shape[2]).mean(axis=0)
    out = crop.copy()
    out[~keep] = mean.astype(crop.dtype)
    return out


class IdentityEncoder:
    """Base class. Subclasses implement ``extract``."""

    name = "base"
    feature_dim = 0

    def extract(
        self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None = None
    ) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError

    def _crop(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        if cv2 is None or frame is None:
            return None
        height, width = frame.shape[:2]
        x, y, w, h = clamp_bbox(bbox, width, height)
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            return None
        return crop

    def _aligned_mask(
        self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None
    ) -> np.ndarray | None:
        """Crop a full-frame boolean mask to the same region as ``_crop``.

        Returns a ``uint8`` (0/255) array matching the crop's H×W (so it can be
        resized alongside the crop and fed to ``cv2.calcHist`` as a mask), or
        ``None`` when no mask is supplied or it does not overlap the box.
        """
        if mask is None or cv2 is None or frame is None:
            return None
        height, width = frame.shape[:2]
        x, y, w, h = clamp_bbox(bbox, width, height)
        sub = mask[y : y + h, x : x + w]
        if sub.size == 0:
            return None
        return (np.asarray(sub).astype("uint8") * 255)


class HsvShapeEncoder(IdentityEncoder):
    """Original encoder: one global 32x32 H-S histogram plus aspect/area shape."""

    name = "hsv_shape"
    feature_dim = 32 * 32 + 2

    def extract(
        self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        crop_mask = self._aligned_mask(frame, bbox, mask)
        height, width = frame.shape[:2]
        _, _, w, h = clamp_bbox(bbox, width, height)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], crop_mask, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        aspect = float(w) / max(1.0, float(h))
        area = float(w * h) / max(1.0, float(width * height))
        shape = np.array([min(4.0, aspect) / 4.0, min(1.0, area * 12.0)], dtype="float32")
        feature = np.concatenate([hist.flatten().astype("float32"), shape])
        return _l2_normalize(feature)


class HsvBlockTextureEncoder(IdentityEncoder):
    """Spatial HSV blocks + gradient-orientation texture + shape.

    The crop is resized to a fixed canvas and split into a ``GRID x GRID`` mesh;
    each cell contributes its own (per-cell normalised) H-S histogram, so the
    feature encodes *where* colours sit on the object rather than a single global
    blob. A magnitude-weighted gradient-orientation histogram adds coarse texture
    so two same-colour objects with different surface detail separate cleanly.
    """

    name = "hsv_block_texture"
    GRID = 2
    H_BINS = 8
    S_BINS = 8
    TEX_BINS = 16
    CANVAS = 64
    feature_dim = GRID * GRID * H_BINS * S_BINS + TEX_BINS + 2

    def extract(
        self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        crop_mask = self._aligned_mask(frame, bbox, mask)
        height, width = frame.shape[:2]
        _, _, w, h = clamp_bbox(bbox, width, height)
        canvas = cv2.resize(crop, (self.CANVAS, self.CANVAS), interpolation=cv2.INTER_AREA)
        # Nearest keeps the mask boolean (no interpolated grey edge) at canvas size.
        canvas_mask = (
            cv2.resize(crop_mask, (self.CANVAS, self.CANVAS), interpolation=cv2.INTER_NEAREST)
            if crop_mask is not None
            else None
        )
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)

        # Per-cell H-S histograms: encode spatial colour layout. Counting only
        # masked (foreground) pixels keeps shadow/road out of the colour signature.
        step = self.CANVAS // self.GRID
        blocks: list[np.ndarray] = []
        for gy in range(self.GRID):
            for gx in range(self.GRID):
                cell = hsv[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
                cell_mask = (
                    canvas_mask[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
                    if canvas_mask is not None
                    else None
                )
                hist = cv2.calcHist([cell], [0, 1], cell_mask, [self.H_BINS, self.S_BINS], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                blocks.append(hist.flatten().astype("float32"))

        # Gradient-orientation texture histogram (HOG-lite), magnitude weighted.
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        angle = (np.arctan2(gy, gx) + np.pi) * (self.TEX_BINS / (2.0 * np.pi))
        bins = np.clip(angle.astype(np.int32), 0, self.TEX_BINS - 1)
        texture = np.bincount(bins.flatten(), weights=magnitude.flatten(), minlength=self.TEX_BINS).astype("float32")
        tex_norm = float(np.linalg.norm(texture))
        if tex_norm > 1e-9:
            texture /= tex_norm

        aspect = float(w) / max(1.0, float(h))
        area = float(w * h) / max(1.0, float(width * height))
        shape = np.array([min(4.0, aspect) / 4.0, min(1.0, area * 12.0)], dtype="float32")

        feature = np.concatenate([*blocks, texture, shape])
        return _l2_normalize(feature)


class DeepEmbeddingEncoder(IdentityEncoder):
    """Deep appearance embedding via an ONNX model (e.g. OSNet) on onnxruntime.

    CPU-capable, so it runs and is testable on a torch-less dev box; on Jetson the
    same ONNX runs through the TensorRT/CUDA execution providers. Loads fully
    guarded: a missing model, missing onnxruntime, or a bad session sets
    ``available = False`` (never raises) so ``build_encoder`` falls back to the
    handcrafted encoder. Returns an L2-normalised vector to match the others.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.available = False
        self.session = None
        self.last_error = ""
        self.feature_dim = 0
        self.name = "deep_onnx"
        deep = config.get("identity", {}).get("deep", {}) if isinstance(config, dict) else {}
        self.model_path = str(deep.get("model_path", ""))
        size = deep.get("input_size", [256, 128])
        self.in_h, self.in_w = int(size[0]), int(size[1])
        providers = list(deep.get("providers", ["CPUExecutionProvider"]))
        if ort is None:
            self.last_error = "onnxruntime not installed"
            return
        if not self.model_path or not Path(self.model_path).is_file():
            self.last_error = f"deep model not found: {self.model_path or '(unset)'}"
            return
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            out_shape = self.session.get_outputs()[0].shape
            last = out_shape[-1] if out_shape else None
            self.feature_dim = int(last) if isinstance(last, int) else int(deep.get("feature_dim", 512))
            self.available = True
            self.name = f"deep_onnx[{Path(self.model_path).stem}]"
        except Exception as exc:  # pragma: no cover - depends on runtime/model
            self.session = None
            self.last_error = f"onnx session init failed: {exc}"

    def extract(
        self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        if not self.available or self.session is None:
            return None
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        crop_mask = self._aligned_mask(frame, bbox, mask)
        if crop_mask is not None:
            crop = _suppress_background(crop, crop_mask)
        blob = cv2.resize(crop, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        blob = (blob - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(blob, (2, 0, 1))[None, ...].astype("float32")  # NCHW
        try:
            out = self.session.run(None, {self.input_name: blob})[0]
        except Exception as exc:  # pragma: no cover - depends on runtime/model
            self.last_error = f"onnx run failed: {exc}"
            return None
        vec = np.asarray(out, dtype="float32").reshape(-1)
        if vec.shape[0] != self.feature_dim:
            self.feature_dim = int(vec.shape[0])
        return _l2_normalize(vec)


class FusedEncoder(IdentityEncoder):
    """Concatenate a fast handcrafted vector with a deep embedding.

    Each sub-vector is scaled by ``sqrt(weight)`` before concatenation, so the
    cosine similarity of two fused vectors equals the weight-blended sum of the
    per-encoder cosine similarities — i.e. ``MemoryBank`` keeps doing a single
    cosine and gets ``w_fast*fast + w_deep*deep`` for free, no change to memory.
    The handcrafted vector already carries colour/texture/shape; the deep vector
    adds instance discrimination to separate genuine look-alikes.
    """

    def __init__(self, fast: IdentityEncoder, deep: DeepEmbeddingEncoder, fast_weight: float, deep_weight: float) -> None:
        self.fast = fast
        self.deep = deep
        self.fast_w = float(fast_weight)
        self.deep_w = float(deep_weight)
        self.feature_dim = int(fast.feature_dim + deep.feature_dim)
        self.name = f"fused({fast.name}+{deep.name})"

    def extract(
        self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None = None
    ) -> np.ndarray | None:
        fast_vec = self.fast.extract(frame, bbox, mask)
        if fast_vec is None:
            return None
        deep_vec = self.deep.extract(frame, bbox, mask)
        if deep_vec is None:
            # Keep the dimension fixed so cosine stays valid: degrade to fast-only
            # by zeroing the deep block (a transient deep failure won't corrupt the bank).
            deep_vec = np.zeros(self.deep.feature_dim, dtype="float32")
        fused = np.concatenate(
            [
                np.sqrt(self.fast_w) * fast_vec.astype("float32"),
                np.sqrt(self.deep_w) * deep_vec.astype("float32"),
            ]
        )
        return _l2_normalize(fused)


def build_encoder(config: dict[str, Any]) -> IdentityEncoder:
    """Construct the configured encoder.

    Reads ``identity.encoder`` (preferred) or legacy ``memory.embedding``. The
    ``deep``/``fused`` options add an ONNX deep embedding and fall back to the
    handcrafted encoder when the model or onnxruntime is unavailable.
    """
    identity_cfg = config.get("identity", {}) if isinstance(config, dict) else {}
    memory_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
    kind = str(identity_cfg.get("encoder", memory_cfg.get("embedding", "hsv_block_texture"))).lower()
    if kind in {"hsv_shape", "hsv", "legacy"}:
        return HsvShapeEncoder()
    if kind in {"deep", "fused", "deep_osnet", "osnet"}:
        deep = DeepEmbeddingEncoder(config)
        if deep.available:
            deep_cfg = identity_cfg.get("deep", {})
            fast_weight = float(deep_cfg.get("fast_weight", 0.45))
            deep_weight = float(deep_cfg.get("deep_weight", 0.45))
            return FusedEncoder(HsvBlockTextureEncoder(), deep, fast_weight, deep_weight)
        # No model / no onnxruntime -> safe fallback (keeps a torch-less dev box running).
        return HsvBlockTextureEncoder()
    return HsvBlockTextureEncoder()
