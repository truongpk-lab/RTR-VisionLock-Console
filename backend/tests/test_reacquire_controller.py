from app.vision.reacquire_controller import ReacquisitionController


class FakeEngine:
    def __init__(self, best):
        self._best = best

    def best(self, frame, candidates):
        return self._best


class FakeMemory:
    def __init__(self, margin):
        self.margin = margin
        self.distractors = 0

    def anchor_score(self, frame, bbox):
        return {"identity_score": 0.9, "negative_margin": self.margin}

    def add_distractor(self, frame, bbox):
        self.distractors += 1


CONFIG = {
    "reacquire": {
        "confirm_frames": 3,
        "confirm_iou_gate": 0.3,
        "confirm_max_gap": 1,
        "learn_distractors": True,
        "distractor_min_score": 0.5,
    },
    "thresholds": {"reacquire_threshold": 0.75},
    "identity": {"min_margin": 0.12},
}

GOOD = {"bbox": [10, 10, 20, 20], "reid_score": 0.9, "score": 0.9}


def _ctrl(best, margin):
    return ReacquisitionController(CONFIG, FakeMemory(margin), FakeEngine(best))


def test_single_good_frame_does_not_relock():
    out = _ctrl(GOOD, margin=0.4).attempt(None, [GOOD])
    assert out.confirmed is False
    assert out.confirming == 1


def test_three_consecutive_frames_confirm_relock():
    ctrl = _ctrl(GOOD, margin=0.4)
    outs = [ctrl.attempt(None, [GOOD]) for _ in range(3)]
    assert outs[-1].confirmed is True
    assert outs[-1].bbox == (10, 10, 20, 20)


def test_lookalike_failing_anchor_is_rejected_and_learned():
    mem = FakeMemory(margin=0.0)  # below min_margin -> look-alike
    ctrl = ReacquisitionController(CONFIG, mem, FakeEngine(GOOD))
    out = ctrl.attempt(None, [GOOD])
    assert out.confirmed is False
    assert mem.distractors == 1  # rejected look-alike taught to the negative bank


def test_below_reid_threshold_never_confirms():
    weak = {"bbox": [10, 10, 20, 20], "reid_score": 0.5, "score": 0.9}
    ctrl = _ctrl(weak, margin=0.4)
    outs = [ctrl.attempt(None, [weak]) for _ in range(5)]
    assert all(not o.confirmed for o in outs)


def test_no_candidates_is_a_miss():
    out = _ctrl(GOOD, margin=0.4).attempt(None, [])
    assert out.confirmed is False
    assert out.confirming == 0
