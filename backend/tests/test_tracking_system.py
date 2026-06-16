"""Integration smoke for the stateful long-term tracking system (no camera).

Drives a TrackingSession with synthetic frames to lock the three guarantees the
upgrade adds: detection is OFF while tracking, global re-detect requires
multi-frame confirmation before re-locking, and the snapshot exposes the new
confidence/re-acquire/memory fields. Runs on a torch-less dev box (backbones fall
back to OpenCV).
"""

import numpy as np

from app.core.session import TrackingSession
from app.core.states import TrackingState

BBOX = (180, 150, 90, 100)


def _frame(x=180, w=90):
    f = np.zeros((480, 640, 3), np.uint8)
    f[150:250, x : x + w] = (60, 200, 240)
    return f


def _locked_session():
    sess = TrackingSession()
    f0 = _frame()
    sess.frame = f0
    sess.learning_bbox = BBOX
    sess._finalize_lock(f0)
    return sess


def test_detection_off_during_locked_tracking():
    sess = _locked_session()
    assert sess.state == TrackingState.LOCKED_TRACKING
    assert sess.confidence.confidence_state == "LOCKED"
    assert sess.candidates == []  # no detector overlays once locked

    calls = []
    sess.proposal.detect = lambda frame: calls.append(1) or []
    for i, x in enumerate([190, 205, 220, 235, 250]):
        sess._frame_count = i + 1
        sess.frame = _frame(x)
        sess._dispatch_frame(sess.frame)  # exercises EKF + camera-motion path too

    assert calls == []  # YOLO/contour detection never ran while tracking
    assert sess.state in {TrackingState.STABLE, TrackingState.UNCERTAIN, TrackingState.LOCKED_TRACKING}


def test_redetect_requires_confirmation_before_relock():
    sess = _locked_session()
    # Simulate a loss and a matching detection at the original location.
    sess.state = TrackingState.SEARCHING
    sess.reacq.reset()
    sess.frame = _frame()
    sess.proposal.detect = lambda frame: [{"bbox": list(BBOX), "score": 0.9}]

    sess._run_reacquire(force=True)
    assert sess.state == TrackingState.SEARCHING  # one confirmation is not enough
    sess._run_reacquire(force=True)
    assert sess.state == TrackingState.SEARCHING  # two is still not enough
    sess._run_reacquire(force=True)
    assert sess.state == TrackingState.LOCKED_TRACKING  # third confirmation re-locks
    assert sess.target_bbox == BBOX


def test_snapshot_exposes_confidence_and_memory_tiers():
    sess = TrackingSession()
    snap = sess.snapshot(include_frame=False)
    tracking = snap["tracking"]
    assert tracking["confidence_state"] in {"LOCKED", "UNCERTAIN", "LOST"}
    assert tracking["reacquire"]["need"] == sess.reacq.confirm_frames
    memory = snap["memory"]
    assert "working_slots" in memory and "anchor_slots" in memory
