from app.core.states import TrackingState


def test_required_states_are_available():
    assert TrackingState.INIT == "INIT"
    assert TrackingState.SEARCHING == "SEARCHING"
    assert TrackingState.REACQUIRED == "REACQUIRED"
