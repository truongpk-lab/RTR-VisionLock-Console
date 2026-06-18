import numpy as np

from app.vision.encoders import (
    DeepEmbeddingEncoder,
    FusedEncoder,
    HsvBlockTextureEncoder,
    HsvShapeEncoder,
    build_encoder,
)
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


def test_deep_encoder_missing_model_is_unavailable_not_crashing():
    enc = DeepEmbeddingEncoder({"identity": {"deep": {"model_path": "models/does_not_exist.onnx"}}})
    assert enc.available is False
    assert enc.last_error
    # extract must degrade gracefully, never raise.
    assert enc.extract(_solid_frame((0, 0, 255)), (80, 80, 40, 40)) is None


def test_build_encoder_fused_falls_back_when_model_absent():
    enc = build_encoder({"identity": {"encoder": "fused", "deep": {"model_path": "models/nope.onnx"}}})
    assert enc.name == "hsv_block_texture"  # safe fallback keeps dev running


def test_fused_encoder_degrades_to_fast_dim_when_deep_returns_none():
    class _DeadDeep:
        name = "deep_stub"
        feature_dim = 8

        def extract(self, frame, bbox, mask=None):
            return None

    fast = HsvBlockTextureEncoder()
    fused = FusedEncoder(fast, _DeadDeep(), fast_weight=0.45, deep_weight=0.45)
    assert fused.feature_dim == fast.feature_dim + 8
    vec = fused.extract(_solid_frame((0, 0, 255)), (80, 80, 40, 40))
    assert vec is not None
    assert vec.shape[0] == fused.feature_dim  # dimension stays fixed for valid cosine
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3


def test_mask_none_matches_default_behaviour():
    """mask=None must reproduce the pre-mask result exactly (safe fallback)."""
    enc = HsvBlockTextureEncoder()
    frame = _solid_frame((0, 0, 255))
    bbox = (80, 80, 40, 40)
    a = enc.extract(frame, bbox)
    b = enc.extract(frame, bbox, mask=None)
    assert a is not None and b is not None
    assert np.allclose(a, b)


def test_mask_excludes_background_from_histogram():
    """A foreground mask keeps shadow/road colour out of the appearance template."""
    enc = HsvShapeEncoder()
    bbox = (60, 60, 80, 80)
    # Box = red object (left half) + grey shadow/road (right half).
    mixed = np.zeros((200, 200, 3), dtype=np.uint8)
    mixed[60:140, 60:100] = (0, 0, 255)
    mixed[60:140, 100:140] = (50, 50, 50)
    # Full-frame boolean mask, True only on the red object.
    mask = np.zeros((200, 200), dtype=bool)
    mask[60:140, 60:100] = True
    # Reference: a box that is entirely the red object.
    pure = np.zeros((200, 200, 3), dtype=np.uint8)
    pure[60:140, 60:140] = (0, 0, 255)

    v_nomask = enc.extract(mixed, bbox)
    v_mask = enc.extract(mixed, bbox, mask=mask)
    v_pure = enc.extract(pure, bbox)
    assert v_nomask is not None and v_mask is not None and v_pure is not None
    # Masking the grey out makes the template match pure-red far better than the
    # mixed crop does -> shadow no longer pollutes the learned colour.
    assert float(np.dot(v_mask, v_pure)) > float(np.dot(v_nomask, v_pure))
    assert float(np.dot(v_nomask, v_pure)) < 0.999  # mixed crop is genuinely worse


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
