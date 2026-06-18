from app.core.metrics import MetricState, clamp, track_score


def test_metric_state_exposes_per_stage_timers():
    metrics = MetricState()
    for field in ("tracker_ms", "reid_ms", "flow_ms", "encode_ms"):
        assert getattr(metrics, field) == 0.0
        assert field in metrics.to_dict()


def test_clamp_bounds_values():
    assert clamp(-1) == 0.0
    assert clamp(2) == 1.0
    assert clamp(0.25) == 0.25


def test_track_score_weighting():
    score = track_score(1.0, 0.8, 0.5, 1.0)
    assert round(score, 2) == 0.85


def test_track_score_negative_penalty_lowers_score():
    base = track_score(1.0, 1.0, 1.0, 1.0)
    penalised = track_score(1.0, 1.0, 1.0, 1.0, negative_penalty=0.3)
    assert base == 1.0
    assert round(penalised, 2) == 0.7  # 1.0 clamped then -0.3


def test_track_score_negative_penalty_defaults_to_noop():
    assert track_score(0.6, 0.6, 0.6, 0.6) == track_score(0.6, 0.6, 0.6, 0.6, negative_penalty=0.0)
