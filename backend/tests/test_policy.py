from app.core.states import TrackingState
from app.vision.policy import TrackingPolicy


def _policy(**thresholds):
    config = {
        "thresholds": {
            "stable_threshold": 0.70,
            "uncertain_threshold": 0.45,
            "lost_frames": 3,
            **thresholds,
        },
    }
    return TrackingPolicy(config)


def test_high_confidence_is_stable():
    policy = _policy()
    decision = policy.update(0.9, ok=True)
    assert decision.state == TrackingState.STABLE
    assert not decision.reacquire


def test_mid_band_is_uncertain_without_reacquire():
    policy = _policy()
    decision = policy.update(0.6, ok=True)
    assert decision.state == TrackingState.UNCERTAIN
    assert not decision.reacquire


def test_long_uncertain_run_never_escalates():
    # A long run inside the uncertain band keeps tracking locally; it must never trip
    # the global re-detect, which is reserved for a genuinely lost target.
    policy = _policy()
    decisions = [policy.update(0.6, ok=True) for _ in range(10)]
    assert all(not d.reacquire for d in decisions)
    assert all(d.state == TrackingState.UNCERTAIN for d in decisions)


def test_consecutive_losses_escalate_to_reacquire():
    policy = _policy()  # lost_frames = 3
    decisions = [policy.update(0.0, ok=False) for _ in range(3)]
    assert not decisions[0].reacquire  # one miss is not yet lost
    assert decisions[0].state == TrackingState.UNCERTAIN
    assert decisions[-1].reacquire is True
    assert decisions[-1].state == TrackingState.LOST


def test_one_good_frame_resets_loss_counter():
    policy = _policy()  # lost_frames = 3
    policy.update(0.0, ok=False)
    policy.update(0.0, ok=False)
    policy.update(0.9, ok=True)  # recovery resets the counter
    follow = [policy.update(0.0, ok=False) for _ in range(2)]
    assert all(not d.reacquire for d in follow)  # needs 3 fresh consecutive losses


def test_sustained_low_identity_escalates_to_lost():
    # Tracker stays ok inside the uncertain band but on the wrong object (low
    # identity): lost_count never increments, so without this gate it would sit in
    # UNCERTAIN forever. After identity_lost_frames it must force LOST + reacquire.
    policy = _policy(identity_lost_frames=3)
    decisions = [policy.update(0.6, ok=True, identity_lost=True) for _ in range(3)]
    assert not decisions[0].reacquire  # still patient
    assert decisions[-1].state == TrackingState.LOST
    assert decisions[-1].reacquire is True


def test_identity_lost_streak_resets_on_recovered_identity():
    policy = _policy(identity_lost_frames=3)
    policy.update(0.6, ok=True, identity_lost=True)
    policy.update(0.6, ok=True, identity_lost=True)
    policy.update(0.6, ok=True, identity_lost=False)  # identity recovered -> reset
    follow = [policy.update(0.6, ok=True, identity_lost=True) for _ in range(2)]
    assert all(not d.reacquire for d in follow)  # needs a fresh full streak


def test_stable_hysteresis_holds_through_dips():
    # Once STABLE, a score dipping below stable_threshold (0.70) but above the exit
    # gate (0.70 - 0.10 = 0.60) must STAY STABLE -- no flapping at the band edge.
    policy = _policy(stable_hysteresis=0.10)
    assert policy.update(0.80, ok=True).state == TrackingState.STABLE  # enter
    assert policy.update(0.65, ok=True).state == TrackingState.STABLE  # held by hysteresis
    assert policy.update(0.62, ok=True).state == TrackingState.STABLE
    # Below the exit gate -> drop to UNCERTAIN.
    assert policy.update(0.55, ok=True).state == TrackingState.UNCERTAIN
    # Re-entry requires the full stable threshold again (not the lowered exit gate).
    assert policy.update(0.65, ok=True).state == TrackingState.UNCERTAIN
    assert policy.update(0.72, ok=True).state == TrackingState.STABLE


def test_no_hysteresis_by_default_is_hard_threshold():
    policy = _policy()  # stable_hysteresis defaults to 0.0
    assert policy.update(0.80, ok=True).state == TrackingState.STABLE
    assert policy.update(0.69, ok=True).state == TrackingState.UNCERTAIN  # hard edge


def test_reset_clears_counters():
    policy = _policy()
    policy.update(0.0, ok=False)
    policy.reset()
    assert policy.lost_count == 0
    assert policy.identity_lost_count == 0


def test_configure_updates_thresholds_without_resetting_counters():
    policy = _policy()
    policy.update(0.0, ok=False)
    before = policy.lost_count
    policy.configure({"thresholds": {"stable_threshold": 0.5}})
    assert policy.stable == 0.5
    assert policy.lost_count == before  # live counters preserved across a config patch
