from app.vision.reacquire import ReAcquireEngine


class FakeMemory:
    def score(self, frame, bbox):
        if bbox[0] == 10:
            return {"identity_score": 0.95, "negative_margin": 0.42}
        return {"identity_score": 0.2, "negative_margin": -0.3}

    def similarity(self, frame, bbox):
        return 0.95 if bbox[0] == 10 else 0.2


class FakeKalman:
    def score_candidate(self, bbox):
        return 0.9 if bbox[0] == 10 else 0.3


def test_reacquire_prefers_identity_over_detector_confidence_only():
    engine = ReAcquireEngine({}, FakeMemory(), FakeKalman())
    candidates = [
        {"id": "Y0", "bbox": [60, 10, 20, 20], "score": 0.99, "mask_quality": 0.3},
        {"id": "Y1", "bbox": [10, 10, 20, 20], "score": 0.65, "mask_quality": 0.9},
    ]

    best = engine.best(None, candidates)

    assert best["id"] == "Y1"
    assert best["similarity"] == 0.95
    assert best["identity_score"] == 0.95
    assert best["negative_margin"] == 0.42
    assert best["mask_quality"] == 0.9
