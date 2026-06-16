from app.vision.policy import ConfidenceManager, TrackMode


def _manager(**tracking):
    config = {
        "thresholds": {"stable_threshold": 0.70, "uncertain_threshold": 0.45, "lost_frames": 3},
        "tracking": {"refind_after": 2, **tracking},
    }
    return ConfidenceManager(config)


def test_locked_allows_memory_and_no_detection():
    gate = _manager().update(0.9, ok=True)
    assert gate.confidence_state == "LOCKED"
    assert gate.allow_memory_update is True
    assert gate.freeze_memory is False
    assert gate.run_detection is False


def test_uncertain_freezes_memory_and_no_detection():
    manager = _manager(refind_after=1)
    gate = manager.update(0.6, ok=True)
    assert gate.confidence_state == "UNCERTAIN"
    assert gate.allow_memory_update is False
    assert gate.freeze_memory is True
    assert gate.run_detection is False


def test_lost_enables_detection_only():
    manager = _manager()  # lost_frames = 3
    gates = [manager.update(0.0, ok=False) for _ in range(3)]
    assert gates[-1].confidence_state == "LOST"
    assert gates[-1].run_detection is True
    assert gates[-1].allow_memory_update is False
    # run_detection must mirror the underlying policy's reacquire trigger exactly.
    assert gates[-1].run_detection == gates[-1].policy.reacquire


def test_mode_passes_through_to_backbone_selection():
    manager = _manager(refind_after=1)
    manager.update(0.6, ok=True)  # -> REFIND
    assert manager.mode == TrackMode.REFIND
    manager.update(0.9, ok=True)  # -> NORMAL
    assert manager.mode == TrackMode.NORMAL


def test_configure_preserves_live_counters():
    manager = _manager()
    manager.update(0.0, ok=False)
    before = manager.policy.lost_count
    manager.configure({"thresholds": {"stable_threshold": 0.5}, "tracking": {}})
    assert manager.policy.stable == 0.5
    assert manager.policy.lost_count == before


def test_reset_returns_to_locked_state():
    manager = _manager(refind_after=1)
    manager.update(0.0, ok=False)
    manager.reset()
    assert manager.confidence_state == "LOCKED"
    assert manager.mode == TrackMode.NORMAL
