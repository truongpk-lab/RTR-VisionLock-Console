import numpy as np

from app.core.states import TrackingState
from app.main import health, session, status
from app.vision.segmenter import SegmentResult


def test_health_endpoint():
    assert health()["status"] == "ok"


def test_status_endpoint_shape():
    data = status()
    assert "state" in data
    assert "metrics" in data
    assert "logs" in data
    assert "proposal" in data


def test_select_target_returns_candidates(monkeypatch):
    session.frame = np.zeros((80, 120, 3), dtype=np.uint8)
    session.state = TrackingState.CAMERA_READY

    monkeypatch.setattr(
        session.proposal,
        "detect",
        lambda frame: [
            {
                "id": "Y0",
                "bbox": [10, 12, 30, 24],
                "score": 0.88,
                "class_id": 1,
                "class_name": "target",
                "source": "yolo",
                "refined": False,
            }
        ],
    )

    data = session.select_target()
    assert data["state"] == "CANDIDATE_TRACKING"
    assert data["candidate_boxes"][0]["source"] == "yolo"
    assert data["candidate_boxes"][0]["track_id"] == "Y0"
    assert "identity_score" in data["candidate_boxes"][0]


def test_pick_target_refines_before_learning(monkeypatch):
    session.frame = np.zeros((80, 120, 3), dtype=np.uint8)
    session.state = TrackingState.CANDIDATE_TRACKING
    session.candidates = [
        {"id": "Y0", "bbox": [10, 12, 30, 24], "score": 0.88, "class_id": 1, "class_name": "target", "source": "yolo", "refined": False},
        {"id": "Y1", "bbox": [42, 12, 30, 24], "score": 0.77, "class_id": 1, "class_name": "target", "source": "yolo", "refined": False},
    ]

    monkeypatch.setattr(
        session.segmenter,
        "refine_box",
        lambda frame, bbox, **kwargs: SegmentResult(bbox=(12, 14, 25, 20), quality=0.93, backend="sam2"),
    )

    data = session.pick_target("Y0")
    assert data["state"] == "LEARNING_TARGET"
    assert data["target_bbox"] == [12, 14, 25, 20]
    assert data["metrics"]["mask_iou"] == 0.93
    assert data["memory"]["negative_slots"] >= 1


def test_segment_endpoint_runs_during_candidate_tracking(monkeypatch):
    """Click-to-segment now coexists with YOLO candidates: a click on open scene
    grabs an object instead of being ignored."""
    session.frame = np.zeros((80, 120, 3), dtype=np.uint8)
    session.state = TrackingState.CANDIDATE_TRACKING
    session.candidates = [{"id": "Y0", "bbox": [10, 12, 30, 24], "score": 0.88, "source": "yolo", "refined": False}]

    called = False

    def fake_segment(*args, **kwargs):
        nonlocal called
        called = True
        return (10, 10, 20, 20)

    monkeypatch.setattr(session.segmenter, "segment_point", fake_segment)
    monkeypatch.setattr(session.segmenter, "refine_box", lambda frame, bbox, **kwargs: None)

    data = session.segment_target({"x": 12, "y": 15})
    assert called is True
    assert data["state"] == "LEARNING_TARGET"


def test_select_box_starts_learning_without_detector(monkeypatch):
    session.frame = np.zeros((80, 120, 3), dtype=np.uint8)
    session.state = TrackingState.CAMERA_READY
    session.candidates = []

    monkeypatch.setattr(session.segmenter, "refine_box", lambda frame, bbox, **kwargs: None)

    data = session.select_box([10, 10, 40, 30])
    assert data["state"] == "LEARNING_TARGET"
    assert data["target_bbox"] == [10, 10, 40, 30]


def test_select_box_rejects_tiny_box():
    session.frame = np.zeros((80, 120, 3), dtype=np.uint8)
    session.state = TrackingState.CAMERA_READY
    data = session.select_box([10, 10, 4, 4])
    assert data["state"] == "CAMERA_READY"
