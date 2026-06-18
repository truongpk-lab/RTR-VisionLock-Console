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


def test_center_gate_keeps_streak_when_iou_zero():
    # Fast target: consecutive boxes barely overlap (IoU~0) but centres are close.
    buf = ConfirmationBuffer(need=2, max_gap=0, iou_gate=0.3, center_gate=1.5)
    buf.push((10, 10, 20, 20), 0.8)
    buf.push((34, 10, 20, 20), 0.8)  # shifted ~24px > box, IoU 0; within 1.5*box
    assert buf.confirmed()


def test_identity_gate_keeps_streak_when_iou_zero():
    # Same target reappeared far away but appearance is a strong match.
    buf = ConfirmationBuffer(need=2, max_gap=0, iou_gate=0.3, identity_gate=0.82)
    buf.push((10, 10, 20, 20), 0.9)
    buf.push((300, 300, 20, 20), 0.9)  # far -> IoU 0, but reid 0.9 >= 0.82
    assert buf.confirmed()


def test_or_gates_off_by_default_preserve_iou_only():
    # With the new gates at their defaults, a far low-IoU jump still restarts.
    buf = ConfirmationBuffer(need=2, max_gap=0, iou_gate=0.3)
    buf.push((10, 10, 20, 20), 0.9)
    buf.push((300, 300, 20, 20), 0.9)
    assert buf.streak == 1
    assert not buf.confirmed()
