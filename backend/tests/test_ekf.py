from types import SimpleNamespace

import numpy as np

from app.vision.ekf import EKFGate


def _center(bbox):
    return bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0


def test_predict_extrapolates_along_velocity():
    g = EKFGate({})
    g.reset((100, 100, 20, 20))  # center (110, 110)
    g.update((110, 100, 20, 20))  # center (120, 110)
    g.update((120, 100, 20, 20))  # center (130, 110)
    cx, _ = _center(g.predict())
    assert cx > 130  # positive velocity carries the prediction past the last box


def test_camera_motion_shifts_prediction():
    g = EKFGate({})
    g.reset((100, 100, 20, 20))
    base_cx, _ = _center(g.predict())
    g.set_camera_motion(SimpleNamespace(ok=True, tx=30.0, ty=0.0, inlier_ratio=0.9))
    shifted_cx, _ = _center(g.predict())
    assert shifted_cx - base_cx == 30  # background pan folded into the prediction
    g.set_camera_motion(SimpleNamespace(ok=False, tx=30.0, ty=0.0, inlier_ratio=0.0))
    nomotion_cx, _ = _center(g.predict())
    assert nomotion_cx == base_cx  # ok=False -> no compensation, falls back to CV


def test_low_inlier_ratio_drops_camera_motion():
    # A moving distractor dominating the flow drives the RANSAC inlier ratio down;
    # the (tx, ty) is then unreliable and must be ignored (constant-velocity only).
    g = EKFGate({})
    g.reset((100, 100, 20, 20))
    base_cx, _ = _center(g.predict())
    g.set_camera_motion(SimpleNamespace(ok=True, tx=30.0, ty=0.0, inlier_ratio=0.2))
    assert _center(g.predict())[0] == base_cx  # below min_inlier_ratio -> rejected
    g.set_camera_motion(SimpleNamespace(ok=True, tx=30.0, ty=0.0, inlier_ratio=0.9))
    assert _center(g.predict())[0] - base_cx == 30  # high ratio -> applied


def test_filter_reduces_center_jitter():
    g = EKFGate({})
    g.reset((100, 100, 20, 20))
    noise = [5, -4, 3, -2, 4, -5, 2, -3, 5, -4]
    measured, filtered = [], []
    for n in noise:
        cx = 110 + n  # center jitters around a constant 110
        g.update((int(cx - 10), 100, 20, 20))
        measured.append(cx)
        filtered.append(float(g.x[0]))
    assert np.var(filtered) < np.var(measured)


def test_contract_parity_with_simple_gate():
    g = EKFGate({})
    assert g.predict() is None  # no estimate before reset/first obs
    error, consistency = g.update((100, 100, 20, 20))
    assert 0.0 <= consistency <= 1.0
    g.update((101, 100, 20, 20))
    error, consistency = g.update((102, 100, 20, 20))
    assert error >= 0.0
    assert 0.0 <= consistency <= 1.0
    assert 0.0 <= g.score_candidate((102, 100, 20, 20)) <= 1.0


def test_score_candidate_without_state_is_one():
    g = EKFGate({})
    assert g.score_candidate((10, 10, 20, 20)) == 1.0
