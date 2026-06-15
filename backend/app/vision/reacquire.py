from __future__ import annotations

from typing import Any

from .kalman import SimpleKalmanGate
from .memory import MemoryBank


class ReAcquireEngine:
    def __init__(self, config: dict[str, Any], memory: MemoryBank, kalman: SimpleKalmanGate) -> None:
        self.config = config
        self.memory = memory
        self.kalman = kalman

    def best(self, frame, candidates: list[dict]) -> dict | None:
        best_candidate = None
        for candidate in candidates:
            bbox = tuple(candidate["bbox"])
            similarity = self.memory.similarity(frame, bbox)
            motion = self.kalman.score_candidate(bbox)
            score = 0.68 * similarity + 0.32 * motion
            enriched = {**candidate, "similarity": round(similarity, 3), "motion": round(motion, 3), "reid_score": round(score, 3)}
            if best_candidate is None or enriched["reid_score"] > best_candidate["reid_score"]:
                best_candidate = enriched
        return best_candidate
