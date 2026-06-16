from app.vision.motion import (
    AdmissionThresholds,
    MaskCandidate,
    MotionWeights,
    memory_admission,
    select_motion_aware_mask,
)


def test_mask_selection_prefers_motion_agreement_over_raw_affinity():
    # A higher-affinity mask sitting far from the Kalman prediction should lose to
    # a slightly lower-affinity mask that lands on the predicted location. This is
    # what keeps the lock on the right instance when a look-alike is nearby.
    predicted = (10, 10, 20, 20)
    candidates = [
        MaskCandidate(bbox=(100, 100, 20, 20), affinity=0.85),  # off-motion
        MaskCandidate(bbox=(10, 10, 20, 20), affinity=0.60),  # on-motion
    ]

    selection = select_motion_aware_mask(candidates, predicted, weights=MotionWeights())

    assert selection is not None
    assert selection.index == 1
    assert selection.bbox == (10, 10, 20, 20)


def test_mask_selection_penalizes_distractor_overlap():
    predicted = (10, 10, 20, 20)
    candidates = [MaskCandidate(bbox=(10, 10, 20, 20), affinity=0.9)]
    negatives = [(12, 12, 20, 20)]

    selection = select_motion_aware_mask(candidates, predicted, negatives=negatives)

    assert selection is not None
    assert selection.distractor_penalty > 0.0


def test_memory_admission_admits_clean_frame():
    decision = memory_admission(positive_sim=0.8, negative_sim=0.2, motion_consistency=0.9, affinity=0.8)
    assert decision.admit is True
    assert decision.reason == "ok"


def test_memory_admission_rejects_near_distractor_and_unstable():
    thresholds = AdmissionThresholds()
    assert memory_admission(0.8, 0.7, 0.9, 0.8, thresholds).reason == "near_distractor"
    assert memory_admission(0.8, 0.2, 0.3, 0.8, thresholds).reason == "unstable_motion"
    assert memory_admission(0.4, 0.2, 0.9, 0.8, thresholds).reason == "low_positive"
    assert memory_admission(0.8, 0.2, 0.9, 0.3, thresholds).reason == "low_affinity"


def test_motion_weights_from_config_reads_samurai_block():
    weights = MotionWeights.from_config({"samurai": {"alpha_kf": 0.4, "distractor_penalty": 0.1}})
    assert weights.alpha_kf == 0.4
    assert weights.distractor == 0.1
