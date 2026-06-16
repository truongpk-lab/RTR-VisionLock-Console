import numpy as np

from app.vision.memory import MemoryBank

F = np.array([1.0, 0.0, 0.0], dtype="float32")  # true target identity
D = np.array([0.0, 1.0, 0.0], dtype="float32")  # orthogonal look-alike / distractor


def test_working_tier_grows_on_cadence_only():
    mem = MemoryBank({"memory": {"working_promote_every": 3}})
    mem.ram.append(F)
    assert len(mem.wrm) == 0
    mem.consolidate(margin=0.0)
    mem.consolidate(margin=0.0)
    assert len(mem.wrm) == 0  # not yet at the promotion cadence
    mem.consolidate(margin=0.0)
    assert len(mem.wrm) == 1  # third admit promotes one working sample


def test_long_term_consolidates_only_above_margin():
    mem = MemoryBank({"memory": {"long_term_min_margin": 0.25}})
    mem.ram.append(F)
    assert mem.consolidate_long_term(margin=0.1) is False
    assert mem.consolidate_long_term(margin=0.4) is True
    assert F in list(mem.drm)


def test_anchor_score_ignores_poisoned_short_term():
    # ram is poisoned with a distractor D; a look-alike query matches ram (high
    # live score) but the anchor (drm+wrm) is the true identity F, so anchor_score
    # stays low and refuses to approve a re-lock.
    mem = MemoryBank({})
    mem.ram.clear(); mem.ram.append(D)
    mem.wrm.clear(); mem.wrm.append(F)
    mem.drm.clear(); mem.drm.append(F)
    mem.extract = lambda frame, bbox: D  # query feature == the look-alike

    live = mem.score(None, None)
    anchor = mem.anchor_score(None, None)

    assert live["positive_similarity"] >= 0.99  # ram match inflates the live score
    assert anchor["positive_similarity"] <= 0.01  # anchor ignores ram -> stays low


def test_anchor_score_zero_when_anchor_empty():
    mem = MemoryBank({})
    mem.ram.append(F)  # short-term present, but anchor (drm+wrm) empty
    assert mem.anchor_score(None, None) == MemoryBank._zero_scores()


def test_add_distractor_grows_negative_bank():
    mem = MemoryBank({})
    mem.extract = lambda frame, bbox: D
    before = len(mem.negative)
    mem.add_distractor(None, None)
    assert len(mem.negative) == before + 1
    assert mem.distractors_added == 1
    assert mem.to_dict()["distractors_added"] == 1
