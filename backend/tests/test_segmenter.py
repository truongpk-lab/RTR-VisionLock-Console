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
