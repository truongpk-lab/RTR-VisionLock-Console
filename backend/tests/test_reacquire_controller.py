from app.vision.reacquire_controller import ReacquisitionController


class FakeEngine:
    def __init__(self, best):
        self._best = best

    def best(self, frame, candidates, lost_age_sec=0.0):
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


def test_moderate_identity_relocks_under_lowered_threshold():
    # A genuine re-appearance with moderate appearance match (reid ~0.65) is blocked
    # forever at the old 0.75 gate but confirms at the tuned 0.62 default, while the
    # anchor-margin gate still guards against distractors.
    cfg = {**CONFIG, "thresholds": {"reacquire_threshold": 0.62}}
    moderate = {"bbox": [10, 10, 20, 20], "reid_score": 0.65, "score": 0.9}
    ctrl = ReacquisitionController(cfg, FakeMemory(margin=0.4), FakeEngine(moderate))
    outs = [ctrl.attempt(None, [moderate]) for _ in range(3)]
    assert outs[-1].confirmed is True


# --- Top-K hypotheses (opt-in) ---

class RankEngine:
    """Engine double that ranks by reid_score (mirrors ReAcquireEngine.rank)."""

    def rank(self, frame, candidates, lost_age_sec=0.0):
        return sorted(candidates, key=lambda c: c.get("reid_score", 0.0), reverse=True)

    def best(self, frame, candidates, lost_age_sec=0.0):
        ranked = self.rank(frame, candidates, lost_age_sec)
        return ranked[0] if ranked else None


TOPK_CONFIG = {**CONFIG, "reacquire": {**CONFIG["reacquire"], "top_k": 2, "confirm_frames": 3}}

A = {"bbox": [10, 10, 20, 20], "reid_score": 0.9, "score": 0.9}
B = {"bbox": [400, 400, 20, 20], "reid_score": 0.85, "score": 0.85}


def test_topk_confirms_a_consistent_runner_up_across_alternating_frames():
    # Two real-looking candidates alternate as top pick each frame. A single buffer
    # would keep resetting; with top_k=2 both streaks accumulate and one confirms.
    ctrl = ReacquisitionController(TOPK_CONFIG, FakeMemory(margin=0.4), RankEngine())
    outs = [ctrl.attempt(None, [A, B]) for _ in range(3)]
    assert outs[-1].confirmed is True
    assert outs[-1].bbox in {(10, 10, 20, 20), (400, 400, 20, 20)}


def test_topk_default_one_keeps_single_buffer_behaviour():
    ctrl = ReacquisitionController(CONFIG, FakeMemory(margin=0.4), RankEngine())
    assert ctrl.top_k == 1
    outs = [ctrl.attempt(None, [GOOD]) for _ in range(3)]
    assert outs[-1].confirmed is True


# --- Instant re-lock fast-path (opt-in) ---

FAST_CONFIG = {
    **CONFIG,
    "reacquire": {**CONFIG["reacquire"], "fast_relock_identity": 0.9, "fast_relock_max_age": 2.0},
}


def test_fast_relock_confirms_on_first_frame_when_certain():
    # reid 0.9 >= fast gate, fresh loss, anchor ok -> re-lock immediately (single path).
    ctrl = ReacquisitionController(FAST_CONFIG, FakeMemory(margin=0.4), FakeEngine(GOOD))
    out = ctrl.attempt(None, [GOOD], lost_age_sec=0.5)
    assert out.confirmed is True
    assert out.bbox == (10, 10, 20, 20)


def test_fast_relock_blocked_after_max_age():
    # Same strong candidate but the loss is old -> must confirm over the streak.
    ctrl = ReacquisitionController(FAST_CONFIG, FakeMemory(margin=0.4), FakeEngine(GOOD))
    out = ctrl.attempt(None, [GOOD], lost_age_sec=5.0)
    assert out.confirmed is False
    assert out.confirming == 1


def test_fast_relock_blocked_below_identity():
    # reid 0.85 < fast gate 0.9 -> no instant re-lock even on a fresh loss.
    weakish = {"bbox": [10, 10, 20, 20], "reid_score": 0.85, "score": 0.9}
    ctrl = ReacquisitionController(FAST_CONFIG, FakeMemory(margin=0.4), FakeEngine(weakish))
    out = ctrl.attempt(None, [weakish], lost_age_sec=0.5)
    assert out.confirmed is False


def test_fast_relock_disabled_by_default():
    # Default config (gate 1.01) never fast-locks: reid 0.9 still needs the streak.
    out = _ctrl(GOOD, margin=0.4).attempt(None, [GOOD], lost_age_sec=0.0)
    assert out.confirmed is False


def test_fast_relock_multi_path_first_frame():
    # top_k=2 path: strongest passer (A, reid 0.9) on a fresh loss re-locks instantly.
    cfg = {**TOPK_CONFIG, "reacquire": {**TOPK_CONFIG["reacquire"], "fast_relock_identity": 0.9, "fast_relock_max_age": 2.0}}
    ctrl = ReacquisitionController(cfg, FakeMemory(margin=0.4), RankEngine())
    out = ctrl.attempt(None, [A, B], lost_age_sec=0.5)
    assert out.confirmed is True
    assert out.bbox == (10, 10, 20, 20)
