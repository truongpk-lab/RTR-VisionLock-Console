import numpy as np

from app.core.session import TrackingSession
from app.vision.segmenter import PromptableSegmenter


def test_filter_by_target_class_narrows_then_falls_back():
    session = TrackingSession()
    detections = [{"class_id": 1}, {"class_id": 2}, {"class_id": 1}]

    # No locked class -> consider everything.
    session.target_class_id = None
    assert session._filter_by_target_class(detections) == detections

    # Locked class -> keep only that label ("find that label again").
    session.target_class_id = 1
    assert session._filter_by_target_class(detections) == [{"class_id": 1}, {"class_id": 1}]

    # Locked class with no match -> fall back to all so it never gets stuck.
    session.target_class_id = 99
    assert session._filter_by_target_class(detections) == detections

    # Switch off -> consider everything regardless of locked class.
    session.target_class_id = 1
    session.config["reacquire"]["match_label"] = False
    assert session._filter_by_target_class(detections) == detections


def test_memory_configure_resizes_banks_without_dropping_features():
    session = TrackingSession()
    for _ in range(3):
        session.memory.ram.append(np.ones(session.memory.feature_dim, dtype=np.float32))

    session.config["memory"]["ram_slots"] = 16
    session.config["identity"]["min_margin"] = 0.2
    session.memory.configure(session.config)

    assert session.memory.ram_slots == 16
    assert session.memory.ram.maxlen == 16
    assert len(session.memory.ram) == 3  # learned features preserved
    assert session.memory.min_margin == 0.2


def test_mask_to_polygon_outlines_a_blob():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True

    polygon = PromptableSegmenter._mask_to_polygon(mask)

    assert polygon is not None
    assert len(polygon) >= 3
    assert all(isinstance(coord, int) for point in polygon for coord in point)
