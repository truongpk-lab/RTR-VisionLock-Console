import numpy as np

from app.vision.segmenter import PromptableSegmenter


def test_segmenter_refine_box_falls_back_when_sam2_unavailable():
    segmenter = PromptableSegmenter(
        {
            "models": {"segmenter": {"enabled": False, "type": "sam2"}},
            "runtime": {"sam_refine_interval": 8},
        }
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[20:55, 30:75] = 255

    result = segmenter.refine_box(frame, (25, 15, 60, 50))

    assert result is not None
    assert result.backend == "grabcut"
    assert len(result.bbox) == 4
    assert 0.0 <= result.quality <= 1.0


class _FakeSam2Predictor:
    """Counts set_image vs predict; returns one filled-rect mask per box prompt."""

    def __init__(self, h: int, w: int) -> None:
        self.set_image_calls = 0
        self.predict_calls = 0
        self._h, self._w = h, w

    def set_image(self, rgb):
        self.set_image_calls += 1

    def predict(self, box=None, point_coords=None, point_labels=None, multimask_output=True):
        self.predict_calls += 1
        x0, y0, x1, y1 = (int(v) for v in box)
        mask = np.zeros((self._h, self._w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return np.array([mask]), np.array([0.9], dtype=np.float32), None


def test_refine_boxes_encodes_image_once():
    seg = PromptableSegmenter({"models": {"segmenter": {"enabled": False}}})
    h, w = 90, 140
    seg.backend = "sam2"
    seg.sam2_predictor = _FakeSam2Predictor(h, w)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    items = [
        {"bbox": (10, 10, 25, 25)},
        {"bbox": (50, 30, 30, 30)},
        {"bbox": (90, 40, 25, 25)},
    ]

    results = seg.refine_boxes(frame, items)

    assert len(results) == 3
    assert seg.sam2_predictor.set_image_calls == 1  # KEY: one ViT encode for the tick
    assert seg.sam2_predictor.predict_calls == 3
    assert all(r is not None and r.backend == "sam2" for r in results)


def test_refine_boxes_falls_back_per_item_when_not_sam2():
    seg = PromptableSegmenter({"models": {"segmenter": {"enabled": False}}})
    assert seg.backend == "grabcut"
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[20:55, 30:75] = 255
    results = seg.refine_boxes(frame, [{"bbox": (25, 15, 60, 50)}])
    assert len(results) == 1
    assert results[0] is not None and results[0].backend == "grabcut"
