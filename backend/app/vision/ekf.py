"""Target-box Kalman gate with camera ego-motion compensation.

Named ``EKFGate`` to match the system's vocabulary, but the model is honestly a
**linear** constant-velocity Kalman filter (CLAUDE.md §1: say what it is). There
is no nonlinear transition/measurement function; the "extended" behaviour that
matters here is the **camera motion control input** injected into the predict
step so the gate tracks the *target's* motion independent of camera shake/pan.

Drop-in replacement for ``SimpleKalmanGate``: identical
``reset``/``predict``/``update``/``score_candidate`` contract and a ``max_error``
attribute (set directly by ``patch_config``). One addition: ``set_camera_motion``
feeds the per-frame background displacement (tx, ty) from the optical-flow
estimator. State vector ``x = [cx, cy, vx, vy, w, h]`` — center + velocity, with
width/height as filtered random-walk states (no size velocity — unneeded).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.metrics import clamp

from .utils import BBox


class EKFGate:
    def __init__(self, config: dict[str, Any]) -> None:
        thresholds = config.get("thresholds", {})
        ekf = config.get("motion", {}).get("ekf", {})
        self.max_error = float(thresholds.get("kalman_max_error", 80))
        q_pos = float(ekf.get("q_pos", 1.0))
        q_vel = float(ekf.get("q_vel", 4.0))
        q_size = float(ekf.get("q_size", 0.5))
        r_pos = float(ekf.get("r_pos", 4.0))
        r_size = float(ekf.get("r_size", 9.0))
        # Constant-velocity transition (dt = 1 frame): cx += vx, cy += vy.
        self._F = np.eye(6, dtype=float)
        self._F[0, 2] = 1.0
        self._F[1, 3] = 1.0
        # Measure center + size; velocity is hidden.
        self._H = np.zeros((4, 6), dtype=float)
        self._H[0, 0] = self._H[1, 1] = self._H[2, 4] = self._H[3, 5] = 1.0
        self._Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_size, q_size]).astype(float)
        self._R = np.diag([r_pos, r_pos, r_size, r_size]).astype(float)
        self._P0 = np.diag([r_pos, r_pos, 4.0 * q_vel, 4.0 * q_vel, r_size, r_size]).astype(float)
        self.reset()

    def reset(self, bbox: BBox | None = None) -> None:
        self._cam = np.zeros(2, dtype=float)
        if bbox is None:
            self.x: np.ndarray | None = None
            self.P: np.ndarray | None = None
            return
        x, y, w, h = bbox
        self.x = np.array([x + w / 2.0, y + h / 2.0, 0.0, 0.0, float(w), float(h)], dtype=float)
        self.P = self._P0.copy()

    def set_camera_motion(self, motion: Any) -> None:
        """Stash the background (tx, ty) consumed by the next predict/update."""
        if motion is not None and getattr(motion, "ok", False):
            self._cam = np.array([float(motion.tx), float(motion.ty)], dtype=float)
        else:
            self._cam = np.zeros(2, dtype=float)

    def _predict_state(self) -> np.ndarray:
        predicted = self._F @ self.x
        predicted[0] += self._cam[0]  # shift prediction by camera ego-motion
        predicted[1] += self._cam[1]
        return predicted

    @staticmethod
    def _to_bbox(state: np.ndarray) -> BBox:
        cx, cy, _, _, w, h = state
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return (int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h)))

    def predict(self) -> BBox | None:
        if self.x is None:
            return None
        return self._to_bbox(self._predict_state())

    def update(self, bbox: BBox) -> tuple[float, float]:
        x, y, w, h = bbox
        z = np.array([x + w / 2.0, y + h / 2.0, float(w), float(h)], dtype=float)
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0, z[2], z[3]], dtype=float)
            self.P = self._P0.copy()
            self._cam = np.zeros(2, dtype=float)
            return 0.0, 1.0
        # Predict (a priori), with camera motion folded in.
        x_pred = self._predict_state()
        p_pred = self._F @ self.P @ self._F.T + self._Q
        # Correct.
        residual = z - self._H @ x_pred
        s = self._H @ p_pred @ self._H.T + self._R
        k = p_pred @ self._H.T @ np.linalg.inv(s)
        self.x = x_pred + k @ residual
        self.P = (np.eye(6) - k @ self._H) @ p_pred
        # Error = how far the (camera-compensated) prediction missed the box center.
        error = float(np.hypot(z[0] - x_pred[0], z[1] - x_pred[1]))
        self._cam = np.zeros(2, dtype=float)  # consumed this frame
        consistency = clamp(1.0 - error / max(1.0, self.max_error))
        return error, consistency

    def score_candidate(self, bbox: BBox) -> float:
        if self.x is None:
            return 1.0
        x_pred = self._predict_state()
        x, y, w, h = bbox
        error = float(np.hypot((x + w / 2.0) - x_pred[0], (y + h / 2.0) - x_pred[1]))
        return clamp(1.0 - error / max(1.0, self.max_error * 2.0))
