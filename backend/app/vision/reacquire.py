from __future__ import annotations

from typing import Any

from .memory import MemoryBank


class ReAcquireEngine:
    # ``kalman`` is any motion gate exposing ``score_candidate`` (EKFGate today).
    def __init__(self, config: dict[str, Any], memory: MemoryBank, kalman: Any) -> None:
        self.config = config
        self.memory = memory
        self.kalman = kalman

    def best(self, frame, candidates: list[dict]) -> dict | None:
        best_candidate = None
        weights = self.config.get("reacquire", {})
        identity_weight = float(weights.get("identity_weight", 0.45))
        motion_weight = float(weights.get("motion_weight", 0.25))
        detector_weight = float(weights.get("detector_weight", 0.15))
        mask_weight = float(weights.get("mask_weight", 0.15))
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
            enriched = {
                **candidate,
                "similarity": round(similarity, 3),
                "identity_score": round(similarity, 3),
                "negative_margin": round(negative_margin, 3),
                "motion": round(motion, 3),
                "motion_score": round(motion, 3),
                "mask_quality": round(mask_quality, 3),
                "reid_score": round(score, 3),
            }
            if best_candidate is None or enriched["reid_score"] > best_candidate["reid_score"]:
                best_candidate = enriched
        return best_candidate
