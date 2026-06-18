from app.vision.reacquire import ReAcquireEngine, dynamic_reacquire_weights


DYN_CFG = {
    "reacquire": {
        "identity_weight": 0.45,
        "motion_weight": 0.25,
        "detector_weight": 0.15,
        "mask_weight": 0.15,
        "lost_short_sec": 1.0,
        "lost_long_sec": 3.0,
        "weights_early": {"identity_weight": 0.40, "motion_weight": 0.30, "detector_weight": 0.15, "mask_weight": 0.15},
        "weights_mid": {"identity_weight": 0.45, "motion_weight": 0.25, "detector_weight": 0.15, "mask_weight": 0.15},
        "weights_late": {"identity_weight": 0.65, "motion_weight": 0.0, "detector_weight": 0.20, "mask_weight": 0.15},
    }
}


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


def test_dynamic_weights_drop_motion_as_loss_ages():
    early = dynamic_reacquire_weights(0.5, DYN_CFG)
    late = dynamic_reacquire_weights(5.0, DYN_CFG)
    assert early["motion_weight"] > late["motion_weight"]
    assert late["motion_weight"] == 0.0
    assert late["identity_weight"] > early["identity_weight"]


def test_dynamic_weights_fall_back_to_static_when_unconfigured():
    w = dynamic_reacquire_weights(5.0, {})
    assert w["identity_weight"] == 0.45  # static default when no tiers configured


class FarTargetMemory:
    # The TRUE target reappeared FAR away after a long loss: high identity there.
    def score(self, frame, bbox):
        if bbox[0] == 500:
            return {"identity_score": 0.9, "negative_margin": 0.4}
        return {"identity_score": 0.2, "negative_margin": -0.2}


class NearBiasMotion:
    # Motion gate favours the NEAR (wrong) box -- the stale Kalman prediction.
    def score_candidate(self, bbox):
        return 0.1 if bbox[0] == 500 else 0.95


def test_late_loss_relocks_far_reappearance_on_identity_not_motion():
    engine = ReAcquireEngine(DYN_CFG, FarTargetMemory(), NearBiasMotion())
    candidates = [
        {"id": "near", "bbox": [50, 10, 20, 20], "score": 0.6, "mask_quality": 0.3},
        {"id": "far", "bbox": [500, 10, 20, 20], "score": 0.6, "mask_quality": 0.3},
    ]
    late = engine.best(None, candidates, lost_age_sec=5.0)
    assert late["id"] == "far"  # motion ~0 at late stage -> identity wins
