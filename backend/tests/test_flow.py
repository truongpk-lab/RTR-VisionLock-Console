import numpy as np
import pytest

from app.vision.flow import CameraMotionEstimator

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

pytestmark = pytest.mark.skipif(cv2 is None, reason="cv2 not available")


def _texture(seed=0, size=200):
    # Coarse random blocks give sharp corners that survive the 0.5 downscale,
    # unlike per-pixel noise which area-averaging smears into flat gray.
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 255, size=(size // 10, size // 10, 3), dtype=np.uint8)
    return cv2.resize(blocks, (size, size), interpolation=cv2.INTER_NEAREST)


def test_recovers_known_translation():
    est = CameraMotionEstimator({"motion": {"camera": {"flow_downscale": 0.5}}})
    frame1 = _texture()
    frame2 = np.roll(frame1, shift=25, axis=1)  # background shifts +25 px in x
    assert est.estimate(frame1, None, 0).ok is False  # first frame: no prev
    motion = est.estimate(frame2, None, 1)
    assert motion.ok is True
    assert abs(motion.tx - 25) < 3
    assert abs(motion.ty) < 3


def test_target_motion_is_masked_out():
    est = CameraMotionEstimator({"motion": {"camera": {"flow_downscale": 0.5}}})
    frame1 = _texture(seed=1)
    frame2 = frame1.copy()  # static background
    bbox = (80, 80, 40, 40)
    # Change only the inside of the target box between frames.
    frame1[85:115, 85:115] = 0
    frame2[85:115, 85:115] = 255
    est.estimate(frame1, bbox, 0)
    motion = est.estimate(frame2, bbox, 1)
    assert motion.ok is True
    assert abs(motion.tx) < 3  # masked target motion does not leak into camera motion
    assert abs(motion.ty) < 3


def test_flat_frame_degrades_gracefully():
    est = CameraMotionEstimator({"motion": {"camera": {"flow_downscale": 0.5}}})
    flat = np.zeros((200, 200, 3), dtype=np.uint8)
    est.estimate(flat, None, 0)
    assert est.estimate(flat, None, 1).ok is False  # no features -> safe fallback


def test_disabled_returns_failed():
    est = CameraMotionEstimator({"motion": {"camera": {"enabled": False}}})
    f1, f2 = _texture(), np.roll(_texture(), 10, axis=1)
    est.estimate(f1, None, 0)
    assert est.estimate(f2, None, 1).ok is False
