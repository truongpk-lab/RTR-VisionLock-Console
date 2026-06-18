from app.core.states import TrackingState
from app.vision.policy import TrackingPolicy, TrackMode


def _policy(**tracking):
    config = {
        "thresholds": {"stable_threshold": 0.70, "uncertain_threshold": 0.45, "lost_frames": 3},
        "tracking": {"refind_after": 2, **tracking},
    }
    return TrackingPolicy(config)


def test_high_confidence_stays_normal_and_stable():
    policy = _policy()
    decision = policy.update(0.9, ok=True)
    assert decision.mode == TrackMode.NORMAL
    assert decision.state == TrackingState.STABLE
    assert not decision.reacquire


def test_mid_band_switches_to_refind_after_patience():
    policy = _policy(refind_after=2)
    first = policy.update(0.6, ok=True)
    assert first.mode == TrackMode.NORMAL  # within patience, still normal
    assert not first.seed_refind
    second = policy.update(0.6, ok=True)
    assert second.mode == TrackMode.REFIND  # patience exhausted -> hand to re-find
    assert second.seed_refind is True
    assert second.state == TrackingState.UNCERTAIN


def test_recovery_reinitializes_normal_tracker():
    policy = _policy(refind_after=1)
    policy.update(0.6, ok=True)  # -> REFIND
    recovered = policy.update(0.9, ok=True)
    assert recovered.mode == TrackMode.NORMAL
    assert recovered.reinit_normal is True
    assert recovered.state == TrackingState.STABLE


def test_consecutive_losses_escalate_to_reacquire():
    policy = _policy()  # lost_frames = 3
    decisions = [policy.update(0.0, ok=False) for _ in range(3)]
    assert decisions[0].mode == TrackMode.REFIND  # slipped out of normal immediately
    assert decisions[0].seed_refind is True
    assert not decisions[0].reacquire
    assert decisions[-1].reacquire is True
    assert decisions[-1].state == TrackingState.LOST


def test_mid_band_does_not_escalate_to_reacquire():
    # A long run inside the uncertain band keeps re-finding; it must never trip the
    # global re-detect, which is reserved for a genuinely lost target.
    policy = _policy(refind_after=1)
    decisions = [policy.update(0.6, ok=True) for _ in range(10)]
    assert all(not d.reacquire for d in decisions)
    assert decisions[-1].mode == TrackMode.REFIND


def test_one_good_frame_resets_loss_counter():
    policy = _policy()  # lost_frames = 3
    policy.update(0.0, ok=False)
    policy.update(0.0, ok=False)
    policy.update(0.9, ok=True)  # recovery resets the counter
    follow = [policy.update(0.0, ok=False) for _ in range(2)]
    assert all(not d.reacquire for d in follow)  # needs 3 fresh consecutive losses


def _policy_with_identity_gate(identity_lost_frames: int) -> TrackingPolicy:
    return TrackingPolicy(
        {
            "thresholds": {
                "stable_threshold": 0.70,
                "uncertain_threshold": 0.45,
                "lost_frames": 3,
                "identity_lost_frames": identity_lost_frames,
            },
            "tracking": {"refind_after": 2},
        }
    )


def test_sustained_low_identity_escalates_to_lost():
    # Tracker stays ok inside the uncertain band but on the wrong object (low
    # identity): lost_count never increments, so without this gate it would sit in
    # UNCERTAIN forever. After identity_lost_frames it must force LOST + reacquire.
    policy = _policy_with_identity_gate(identity_lost_frames=3)
    decisions = [policy.update(0.6, ok=True, identity_lost=True) for _ in range(3)]
    assert not decisions[0].reacquire  # still patient
    assert decisions[-1].state == TrackingState.LOST
    assert decisions[-1].reacquire is True


def test_identity_lost_streak_resets_on_recovered_identity():
    policy = _policy_with_identity_gate(identity_lost_frames=3)
    policy.update(0.6, ok=True, identity_lost=True)
    policy.update(0.6, ok=True, identity_lost=True)
    policy.update(0.6, ok=True, identity_lost=False)  # identity recovered -> reset
    follow = [policy.update(0.6, ok=True, identity_lost=True) for _ in range(2)]
    assert all(not d.reacquire for d in follow)  # needs a fresh full streak


def test_reset_returns_to_normal_mode():
    policy = _policy(refind_after=1)
    policy.update(0.6, ok=True)
    assert policy.mode == TrackMode.REFIND
    policy.reset()
    assert policy.mode == TrackMode.NORMAL
    assert policy.lost_count == 0


def test_configure_updates_thresholds_without_resetting_counters():
    policy = _policy()
    policy.update(0.0, ok=False)
    before = policy.lost_count
    policy.configure({"thresholds": {"stable_threshold": 0.5}, "tracking": {}})
    assert policy.stable == 0.5
    assert policy.lost_count == before  # live counters preserved across a config patch
