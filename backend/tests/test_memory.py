"""Per-tick feature cache: one encode per (frame, bbox), without merging the
ram+drm (score) and drm+wrm (anchor_score) identities."""

import numpy as np

from app.vision.memory import MemoryBank


def _feat(idx: int, dim: int = 4) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    v[idx % dim] = 1.0
    return v


def _bank() -> MemoryBank:
    return MemoryBank({})


def test_feature_cache_encodes_once_per_tick():
    mem = _bank()
    mem.ram.append(_feat(0))
    mem.drm.append(_feat(1))
    mem.wrm.append(_feat(2))
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    bbox = (5, 5, 10, 10)

    calls = {"n": 0}

    def fake_extract(f, b, mask=None):
        calls["n"] += 1
        return _feat(0)

    mem.extract = fake_extract

    mem.begin_tick(frame)
    mem.score(frame, bbox)
    mem.anchor_score(frame, bbox)
    mem.score(frame, bbox)
    assert calls["n"] == 1  # score+anchor+score share ONE encode for (frame, bbox)

    mem.score(frame, (9, 9, 10, 10))  # a different bbox must encode again
    assert calls["n"] == 2

    mem.end_tick()
    mem.score(frame, bbox)  # outside the tick the cache is gone -> re-encode
    assert calls["n"] == 3


def test_cache_does_not_merge_score_and_anchor():
    mem = _bank()
    mem.ram.append(_feat(0))   # score sees ram+drm = {e0, e1}
    mem.drm.append(_feat(1))
    mem.wrm.append(_feat(2))   # anchor sees drm+wrm = {e1, e2}
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    bbox = (5, 5, 10, 10)
    mem.extract = lambda f, b, mask=None: _feat(0)  # feature == e0

    mem.begin_tick(frame)
    score = mem.score(frame, bbox)
    anchor = mem.anchor_score(frame, bbox)
    mem.end_tick()

    # Same cached feature, but different positive sets -> different similarity.
    assert score["positive_similarity"] == 1.0   # e0 matches ram's e0
    assert anchor["positive_similarity"] == 0.0   # e0 not in drm/wrm
    assert score["positive_similarity"] != anchor["positive_similarity"]
