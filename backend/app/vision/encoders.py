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

from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .utils import BBox, clamp_bbox


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


class IdentityEncoder:
    """Base class. Subclasses implement ``extract``."""

    name = "base"
    feature_dim = 0

    def extract(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:  # pragma: no cover - interface
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


class HsvShapeEncoder(IdentityEncoder):
    """Original encoder: one global 32x32 H-S histogram plus aspect/area shape."""

    name = "hsv_shape"
    feature_dim = 32 * 32 + 2

    def extract(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        height, width = frame.shape[:2]
        _, _, w, h = clamp_bbox(bbox, width, height)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
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

    def extract(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        crop = self._crop(frame, bbox)
        if crop is None:
            return None
        height, width = frame.shape[:2]
        _, _, w, h = clamp_bbox(bbox, width, height)
        canvas = cv2.resize(crop, (self.CANVAS, self.CANVAS), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)

        # Per-cell H-S histograms: encode spatial colour layout.
        step = self.CANVAS // self.GRID
        blocks: list[np.ndarray] = []
        for gy in range(self.GRID):
            for gx in range(self.GRID):
                cell = hsv[gy * step : (gy + 1) * step, gx * step : (gx + 1) * step]
                hist = cv2.calcHist([cell], [0, 1], None, [self.H_BINS, self.S_BINS], [0, 180, 0, 256])
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


def build_encoder(config: dict[str, Any]) -> IdentityEncoder:
    """Construct the configured encoder.

    Reads ``identity.encoder`` (preferred) or legacy ``memory.embedding``.
    """
    identity_cfg = config.get("identity", {}) if isinstance(config, dict) else {}
    memory_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
    kind = str(identity_cfg.get("encoder", memory_cfg.get("embedding", "hsv_block_texture"))).lower()
    if kind in {"hsv_shape", "hsv", "legacy"}:
        return HsvShapeEncoder()
    return HsvBlockTextureEncoder()
