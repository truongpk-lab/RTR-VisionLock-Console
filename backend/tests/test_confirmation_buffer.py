from app.vision.reacquire_controller import ConfirmationBuffer

BOX = (10, 10, 20, 20)


def test_three_consistent_pushes_confirm():
    buf = ConfirmationBuffer(need=3, max_gap=1, iou_gate=0.3)
    buf.push(BOX, 0.8)
    assert not buf.confirmed()
    buf.push(BOX, 0.8)
    assert not buf.confirmed()
    buf.push(BOX, 0.9)
    assert buf.confirmed()
    assert buf.best_bbox() == BOX  # highest-scoring confirmed box


def test_miss_within_gap_keeps_streak():
    buf = ConfirmationBuffer(need=3, max_gap=1, iou_gate=0.3)
    buf.push(BOX, 0.8)
    buf.push(BOX, 0.8)
    buf.miss()  # gap=1 <= max_gap -> streak tolerated
    buf.push(BOX, 0.8)
    assert buf.confirmed()


def test_miss_beyond_gap_resets():
    buf = ConfirmationBuffer(need=2, max_gap=1, iou_gate=0.3)
    buf.push(BOX, 0.8)
    buf.miss()
    buf.miss()  # gap=2 > max_gap -> reset
    assert buf.streak == 0
    buf.push(BOX, 0.8)
    assert not buf.confirmed()


def test_spatial_jump_restarts_streak():
    buf = ConfirmationBuffer(need=2, max_gap=0, iou_gate=0.3)
    buf.push((10, 10, 20, 20), 0.8)
    buf.push((300, 300, 20, 20), 0.9)  # far away -> iou 0 -> restart, not confirm
    assert buf.streak == 1
    assert not buf.confirmed()


def test_reset_clears_state():
    buf = ConfirmationBuffer(need=2, max_gap=1, iou_gate=0.3)
    buf.push(BOX, 0.8)
    buf.reset()
    assert buf.streak == 0
    assert buf.best_bbox() is None
