from __future__ import annotations

from typing import Any

from .memory import MemoryBank


def dynamic_reacquire_weights(lost_age_sec: float, cfg: dict[str, Any]) -> dict[str, float]:
    """Re-acquire scoring weights that shift with how long the target has been lost.

    Just lost -> still trust motion (the target is near the last prediction).
    Lost long -> motion is meaningless (it can reappear anywhere), so lean on
    appearance identity. This is what lets a far-away re-appearance be re-locked
    instead of being penalised for being far from the stale Kalman prediction.
    Falls back to the static ``reacquire`` weights when a tier is not configured.
    """
    reacq = cfg.get("reacquire", {})
    static = {
        "identity_weight": float(reacq.get("identity_weight", 0.45)),
        "motion_weight": float(reacq.get("motion_weight", 0.25)),
        "detector_weight": float(reacq.get("detector_weight", 0.15)),
        "mask_weight": float(reacq.get("mask_weight", 0.15)),
    }
    short = float(reacq.get("lost_short_sec", 1.0))
    long_lost = float(reacq.get("lost_long_sec", 3.0))
    if lost_age_sec < short:
        tier = reacq.get("weights_early")
    elif lost_age_sec < long_lost:
        tier = reacq.get("weights_mid")
    else:
        tier = reacq.get("weights_late")
    if not isinstance(tier, dict):
        return static
    return {name: float(tier.get(name, static[name])) for name in static}


class ReAcquireEngine:
    # ``kalman`` is any motion gate exposing ``score_candidate`` (EKFGate today).
    def __init__(self, config: dict[str, Any], memory: MemoryBank, kalman: Any) -> None:
        self.config = config
        self.memory = memory
        self.kalman = kalman

    def rank(self, frame, candidates: list[dict], lost_age_sec: float = 0.0) -> list[dict]:
        """Score every candidate and return them sorted by reid_score (desc).

        lost_age_sec > 0 -> shift weights by how long we've been lost (far
        re-appearances need identity, not motion). 0 keeps the static weights so
        callers that don't track loss age (and existing tests) are unaffected.
        """
        if lost_age_sec > 0.0:
            weights = dynamic_reacquire_weights(lost_age_sec, self.config)
        else:
            weights = self.config.get("reacquire", {})
        identity_weight = float(weights.get("identity_weight", 0.45))
        motion_weight = float(weights.get("motion_weight", 0.25))
        detector_weight = float(weights.get("detector_weight", 0.15))
        mask_weight = float(weights.get("mask_weight", 0.15))
        enriched_all: list[dict] = []
        for candidate in candidates:
            bbox = tuple(candidate["bbox"])
            if hasattr(self.memory, "score"):
                identity = self.memory.score(frame, bbox)
                similarity = float(identity.get("identity_score", 0.0))
                negative_margin = float(identity.get("negative_margin", 0.0))
            else:
                similarity = self.memory.similarity(frame, bbox)
                negative_margin = 0.0
            motion = self.kalman.score_candidate(bbox)
            detector = float(candidate.get("score", 0.0))
            mask_quality = float(candidate.get("mask_quality", 0.0))
            score = (
                identity_weight * similarity
                + motion_weight * motion
                + detector_weight * detector
                + mask_weight * mask_quality
            )
            enriched_all.append(
                {
                    **candidate,
                    "similarity": round(similarity, 3),
                    "identity_score": round(similarity, 3),
                    "negative_margin": round(negative_margin, 3),
                    "motion": round(motion, 3),
                    "motion_score": round(motion, 3),
                    "mask_quality": round(mask_quality, 3),
                    "reid_score": round(score, 3),
                }
            )
        enriched_all.sort(key=lambda c: c["reid_score"], reverse=True)
        return enriched_all

    def best(self, frame, candidates: list[dict], lost_age_sec: float = 0.0) -> dict | None:
        ranked = self.rank(frame, candidates, lost_age_sec)
        return ranked[0] if ranked else None
