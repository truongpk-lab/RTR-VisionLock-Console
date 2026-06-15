from app.core.metrics import clamp, track_score


def test_clamp_bounds_values():
    assert clamp(-1) == 0.0
    assert clamp(2) == 1.0
    assert clamp(0.25) == 0.25


def test_track_score_weighting():
    score = track_score(1.0, 0.8, 0.5, 1.0)
    assert round(score, 2) == 0.85
