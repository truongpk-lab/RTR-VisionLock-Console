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


def test_searching_clears_frozen_target_box():
    """Lost target must not leave a frozen box: entering re-acquire clears
    target_bbox/kalman_bbox so the snapshot (and UI) hides the stale overlay
    until a confirmed re-lock re-populates it."""
    sess = _locked_session()
    assert sess.target_bbox == BBOX  # locked box is present before loss
    sess.state = TrackingState.SEARCHING
    sess.reacq.reset()
    sess.frame = _frame()
    sess.proposal.detect = lambda frame: []  # nothing to confirm -> stays searching

    sess._run_reacquire(force=True)
    assert sess.state == TrackingState.SEARCHING
    assert sess.target_bbox is None
    assert sess.kalman_bbox is None
    snap = sess.snapshot(include_frame=False)
    assert snap["target_bbox"] is None
    assert snap["kalman_bbox"] is None


def test_snapshot_exposes_confidence_and_memory_tiers():
    sess = TrackingSession()
    snap = sess.snapshot(include_frame=False)
    tracking = snap["tracking"]
    assert tracking["confidence_state"] in {"LOCKED", "UNCERTAIN", "LOST"}
    assert tracking["reacquire"]["need"] == sess.reacq.confirm_frames
    # Fallback must be observable so the UI can warn the operator (A1).
    assert isinstance(tracking["tracker_fallback"], bool)
    memory = snap["memory"]
    assert "working_slots" in memory and "anchor_slots" in memory


def test_snapshot_exposes_debug_block():
    sess = _locked_session()
    sess._frame_count = 1
    sess.frame = _frame(190)
    sess._update_tracking(sess.frame)
    debug = sess.snapshot(include_frame=False)["debug"]
    for key in (
        "tracker_backend",
        "tracker_fallback",
        "proposal_source",
        "lost_age_sec",
        "negative_similarity",
        "positive_negative_margin",
        "reacquire_score",
        "ego_motion_ok",
    ):
        assert key in debug
    assert debug["proposal_source"] == "tracker_normal"
    assert isinstance(debug["ego_motion_ok"], bool)


def test_fallback_confidence_tracks_identity_not_just_jitter():
    """A4a: OpenCV fallback confidence follows appearance identity, so SMOOTH
    drift (low identity but low jitter) is no longer read as a solid lock."""

    def run(identity_score):
        sess = _locked_session()
        assert sess.tracker.is_fallback  # torch-less dev box -> OpenCV fallback
        sess.memory.score = lambda frame, bbox, s=identity_score: {
            "positive_similarity": s,
            "negative_similarity": 0.1,
            "identity_score": s,
            "negative_margin": s - 0.1,
        }
        sess._frame_count = 1
        sess.frame = _frame(185)  # tiny move -> low jitter (stability stays high)
        sess._update_tracking(sess.frame)
        return sess.metrics.confidence

    high = run(0.95)
    low = run(0.20)
    assert high > low  # identity, not jitter alone, drives fallback confidence
    assert low <= 0.75  # fallback confidence is capped


def test_fallback_drift_drops_state_and_freezes_memory():
    """A4c + A2: with the OpenCV fallback running, low identity (drift) drops the
    state out of LOCKED and stops the bank from learning the drifted crop."""
    sess = _locked_session()
    assert sess.tracker.is_fallback
    sess.memory.score = lambda frame, bbox: {
        "positive_similarity": 0.2,
        "negative_similarity": 0.1,
        "identity_score": 0.2,
        "negative_margin": 0.1,
    }
    before = sess.memory.admitted_count
    for i, x in enumerate([185, 195, 205, 215, 225, 235]):
        sess._frame_count = i + 1
        sess.frame = _frame(x)
        sess._update_tracking(sess.frame)
    assert sess.confidence.confidence_state != "LOCKED"  # drift detected, not hidden
    assert sess.memory.admitted_count == before  # drifted crop never poisons memory


def test_reid_cadence_skips_deep_between_intervals():
    """P3: with reid_interval>1 the deep identity score runs only on cadence frames
    (reid_on_uncertain off here to isolate the cadence from state dynamics)."""
    sess = _locked_session()
    sess.config.setdefault("identity", {})
    sess.config["identity"]["reid_interval"] = 3
    sess.config["identity"]["reid_on_uncertain"] = False
    cached = {"positive_similarity": 0.9, "negative_similarity": 0.1,
              "identity_score": 0.9, "negative_margin": 0.8}
    calls = []
    sess.memory.score = lambda frame, bbox: (calls.append(1), dict(cached))[1]
    sess._last_identity = dict(cached)  # seed the cache so the first frame can skip
    for i in range(1, 10):  # frame_count 1..9
        sess._frame_count = i
        sess.frame = _frame(180)
        sess._update_tracking(sess.frame)
    # Deep score runs only at frame_count 3, 6, 9 -> 3 of 9 frames.
    assert len(calls) == 3
