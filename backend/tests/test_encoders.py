import numpy as np

from app.vision.encoders import HsvBlockTextureEncoder, HsvShapeEncoder, build_encoder
from app.vision.memory import MemoryBank


def _solid_frame(color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[80:120, 80:120] = color  # BGR block in the centre
    return frame


def test_build_encoder_default_is_block_texture_and_hsv_shape_selectable():
    assert build_encoder({}).name == "hsv_block_texture"
    assert build_encoder({"identity": {"encoder": "hsv_shape"}}).name == "hsv_shape"


def test_block_texture_separates_different_colours():
    encoder = HsvBlockTextureEncoder()
    bbox = (80, 80, 40, 40)
    red = encoder.extract(_solid_frame((0, 0, 255)), bbox)
    blue = encoder.extract(_solid_frame((255, 0, 0)), bbox)
    red2 = encoder.extract(_solid_frame((0, 0, 255)), bbox)

    assert red is not None and blue is not None and red2 is not None
    assert red.shape[0] == encoder.feature_dim
    same = float(np.dot(red, red2))
    different = float(np.dot(red, blue))
    assert same > 0.95
    assert different < same - 0.3


def test_hsv_shape_encoder_dim_matches():
    feature = HsvShapeEncoder().extract(_solid_frame((0, 255, 0)), (80, 80, 40, 40))
    assert feature is not None
    assert feature.shape[0] == HsvShapeEncoder.feature_dim


def test_memory_consider_update_rejects_distractor_frame():
    bank = MemoryBank({"identity": {"encoder": "hsv_block_texture"}})
    bbox = (80, 80, 40, 40)
    # Lock onto a red target; register a blue distractor as negative.
    assert bank.initialize(_solid_frame((0, 0, 255)), bbox)
    bank.update_negative(_solid_frame((255, 0, 0)), bbox)

    # A frame that actually looks like the blue distractor must NOT enter memory.
    decision = bank.consider_update(_solid_frame((255, 0, 0)), bbox, motion_consistency=0.9, affinity=0.9)
    assert decision.admit is False

    # A clean red frame on stable motion is admitted.
    ram_before = len(bank.ram)
    admitted = bank.consider_update(_solid_frame((0, 0, 255)), bbox, motion_consistency=0.9, affinity=0.9)
    assert admitted.admit is True
    assert len(bank.ram) == ram_before + 1
