import numpy as np

from app.vision.proposal import YoloProposalDetector


def test_yolo_candidate_builder_clamps_and_preserves_schema():
    detector = YoloProposalDetector(
        {
            "models": {
                "proposal": {
                    "enabled": False,
                    "type": "yolo11_trt",
                    "class_names": ["target"],
                }
            },
            "runtime": {"max_candidates": 5},
        }
    )

    candidates = detector._build_candidates(
        [(-5.0, 4.0, 25.0, 34.0, 0.91, 0)],
        width=20,
        height=30,
    )

    assert candidates[0]["id"] == "Y0"
    assert candidates[0]["track_id"] == "Y0"
    assert candidates[0]["bbox"] == [0, 4, 20, 26]
    assert candidates[0]["score"] == 0.91
    assert candidates[0]["class_name"] == "target"
    assert candidates[0]["source"] == "yolo"
    assert candidates[0]["mask_quality"] == 0.0
    assert candidates[0]["identity_score"] == 0.0
    assert candidates[0]["negative_margin"] == 0.0
    assert candidates[0]["motion_score"] == 0.0
    assert candidates[0]["is_distractor"] is False


def test_yolo_candidate_builder_handles_empty_detections():
    detector = YoloProposalDetector({"models": {"proposal": {"enabled": False}}, "runtime": {}})

    assert detector._build_candidates([], width=1280, height=720) == []


def test_candidate_mask_polygon_none_without_segmentation():
    detector = YoloProposalDetector({"models": {"proposal": {"enabled": False}}, "runtime": {"max_candidates": 5}})

    candidates = detector._build_candidates([(0.0, 0.0, 10.0, 10.0, 0.8, 0)], width=100, height=100)

    assert candidates[0]["mask_polygon"] is None


def test_candidate_mask_polygon_attached_and_simplified():
    detector = YoloProposalDetector({"models": {"proposal": {"enabled": False}}, "runtime": {"max_candidates": 5}})
    polygon = [[float(x), float(y)] for x, y in [(0, 0), (10, 0), (10, 10), (0, 10), (0, 5)]]

    candidates = detector._build_candidates(
        [(0.0, 0.0, 10.0, 10.0, 0.8, 0)], width=100, height=100, polygons=[polygon]
    )

    mask = candidates[0]["mask_polygon"]
    assert mask is not None and len(mask) >= 3
    assert all(isinstance(coord, int) for point in mask for coord in point)


def test_yolo_detector_falls_back_without_runtime_or_engine():
    detector = YoloProposalDetector(
        {
            "models": {
                "proposal": {
                    "enabled": True,
                    "type": "yolo11_trt",
                    "path": "models/missing.engine",
                }
            },
            "runtime": {"max_candidates": 3},
        }
    )
    frame = np.zeros((64, 96, 3), dtype=np.uint8)

    candidates = detector.detect(frame)

    assert candidates
    assert candidates[0]["source"] == "opencv"
