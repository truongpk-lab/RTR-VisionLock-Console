"""Sparse optical-flow camera ego-motion estimator.

Estimates the global background displacement between consecutive frames so the
:class:`~app.vision.ekf.EKFGate` can subtract camera shake/pan and keep tracking
the *target's* own motion. The target box is masked out of the feature set so the
estimate reflects the scene, not the object.

Cheap by construction (the >=60fps budget): gray + downscale, capped
``goodFeaturesToTrack``, ``calcOpticalFlowPyrLK``, RANSAC ``estimateAffinePartial2D``.
Costly only if misconfigured (full-res / unbounded features / dense flow) — hence
the ``flow_downscale`` / ``max_features`` / ``flow_interval`` knobs. Pure OpenCV
CPU; degrades to ``ok=False`` (gate falls back to plain CV predict) on any failure
or when cv2 is unavailable, so a torch/GPU-less dev box still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .utils import BBox, clamp_bbox


@dataclass
class CameraMotion:
    tx: float
    ty: float
    scale: float
    inlier_ratio: float
    ok: bool


_FAILED = CameraMotion(0.0, 0.0, 1.0, 0.0, False)


class CameraMotionEstimator:
    def __init__(self, config: dict[str, Any]) -> None:
        cam = config.get("motion", {}).get("camera", {})
        self.enabled = bool(cam.get("enabled", True))
        self.flow_downscale = float(cam.get("flow_downscale", 0.5))
        self.max_features = int(cam.get("max_features", 200))
        self.flow_interval = max(1, int(cam.get("flow_interval", 1)))
        self.feature_quality = float(cam.get("feature_quality", 0.01))
        self.feature_min_distance = int(cam.get("feature_min_distance", 8))
        self.target_mask_margin = float(cam.get("target_mask_margin", 0.2))
        self.min_inliers = int(cam.get("min_inliers", 12))
        self.lk_win = int(cam.get("lk_win", 21))
        self.reset()

    def reset(self) -> None:
        self._prev_gray: np.ndarray | None = None
        self._last = _FAILED

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.flow_downscale != 1.0:
            gray = cv2.resize(
                gray, None, fx=self.flow_downscale, fy=self.flow_downscale, interpolation=cv2.INTER_AREA
            )
        return gray

    def _target_mask(self, shape: tuple[int, int], target_bbox: BBox | None) -> np.ndarray | None:
        if target_bbox is None:
            return None
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        s = self.flow_downscale
        bx, by, bw, bh = target_bbox
        # Scale the target box into the downscaled frame and dilate by the margin.
        mx = int(bw * self.target_mask_margin)
        my = int(bh * self.target_mask_margin)
        x, y, ww, hh = clamp_bbox(
            (int((bx - mx) * s), int((by - my) * s), int((bw + 2 * mx) * s), int((bh + 2 * my) * s)), w, h
        )
        mask[y : y + hh, x : x + ww] = 0  # exclude the target from background flow
        return mask

    def estimate(self, frame: np.ndarray, target_bbox: BBox | None, frame_index: int) -> CameraMotion:
        if cv2 is None or not self.enabled or frame is None:
            return _FAILED
        try:
            gray = self._prep(frame)
        except Exception:  # pragma: no cover - malformed frame
            return _FAILED
        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            return _FAILED
        if self.flow_interval > 1 and frame_index % self.flow_interval != 0:
            return self._last
        self._last = self._estimate(prev, gray, target_bbox)
        return self._last

    def _estimate(self, prev: np.ndarray, cur: np.ndarray, target_bbox: BBox | None) -> CameraMotion:
        try:
            mask = self._target_mask(prev.shape, target_bbox)
            pts = cv2.goodFeaturesToTrack(
                prev,
                maxCorners=self.max_features,
                qualityLevel=self.feature_quality,
                minDistance=self.feature_min_distance,
                mask=mask,
            )
            if pts is None or len(pts) < self.min_inliers:
                return _FAILED
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev, cur, pts, None, winSize=(self.lk_win, self.lk_win), maxLevel=2
            )
            if nxt is None or status is None:
                return _FAILED
            keep = status.flatten() == 1
            good_prev = pts[keep]
            good_next = nxt[keep]
            if len(good_prev) < self.min_inliers:
                return _FAILED
            matrix, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
            if matrix is None:
                return _FAILED
            n_inliers = int(inliers.sum()) if inliers is not None else len(good_prev)
            if n_inliers < self.min_inliers:
                return _FAILED
            inv = 1.0 / self.flow_downscale if self.flow_downscale else 1.0
            tx = float(matrix[0, 2]) * inv
            ty = float(matrix[1, 2]) * inv
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0])) or 1.0
            inlier_ratio = n_inliers / max(1, len(good_prev))
            return CameraMotion(tx=tx, ty=ty, scale=scale, inlier_ratio=inlier_ratio, ok=True)
        except Exception:  # pragma: no cover - depends on cv2 build
            return _FAILED
