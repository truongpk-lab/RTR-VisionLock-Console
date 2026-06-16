"""Global re-detection with multi-frame confirmation before re-lock.

When the target is LOST, the session runs the full-frame detector at low frequency
and feeds the candidates here. This controller does NOT re-lock on the first
plausible frame (the old behaviour, which flapped on a single false ReID); it
requires a candidate to be confirmed over several consecutive, spatially
consistent frames AND to match the *long-term anchor* (not the volatile
short-term memory) before declaring a re-lock.

Pure logic over plain dicts/bboxes — no cv2/torch — so it is unit-testable with
fake memory/engine collaborators (see tests/test_reacquire_controller.py and the
fakes in tests/test_reacquire.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .reacquire import ReAcquireEngine
from .utils import BBox, bbox_iou


class ConfirmationBuffer:
    """Counts consecutive, spatially-consistent confirmations of a candidate.

    ``need`` consecutive consistent hits confirm a re-lock. A short dropout is
    tolerated up to ``max_gap`` misses; a longer gap or a large spatial jump
    resets the streak so a flickering false match can never accumulate.
    """

    def __init__(self, need: int, max_gap: int, iou_gate: float) -> None:
        self.need = max(1, int(need))
        self.max_gap = max(0, int(max_gap))
        self.iou_gate = float(iou_gate)
        self.reset()

    def reset(self) -> None:
        self.streak = 0
        self.gap = 0
        self.last_bbox: BBox | None = None
        self._best_bbox: BBox | None = None
        self._best_score = -1.0

    def push(self, bbox: BBox, reid_score: float) -> None:
        bbox = tuple(int(v) for v in bbox)
        consistent = self.last_bbox is None or bbox_iou(self.last_bbox, bbox) >= self.iou_gate
        if consistent:
            self.streak += 1
        else:
            # Spatial jump: this is a different location, restart on the new one.
            self.streak = 1
            self._best_bbox = None
            self._best_score = -1.0
        self.gap = 0
        self.last_bbox = bbox
        if reid_score > self._best_score:
            self._best_score = float(reid_score)
            self._best_bbox = bbox

    def miss(self) -> None:
        self.gap += 1
        if self.gap > self.max_gap:
            self.reset()

    def confirmed(self) -> bool:
        return self.streak >= self.need

    def best_bbox(self) -> BBox | None:
        return self._best_bbox


@dataclass
class ReacqOutcome:
    confirmed: bool
    bbox: BBox | None
    reid_score: float
    candidate: dict | None
    confirming: int  # current confirmation streak
    need: int


class ReacquisitionController:
    def __init__(self, config: dict[str, Any], memory: Any, engine: ReAcquireEngine) -> None:
        self.memory = memory
        self.engine = engine
        self.configure(config)
        self.buffer = ConfirmationBuffer(self.confirm_frames, self.confirm_max_gap, self.confirm_iou_gate)

    def configure(self, config: dict[str, Any]) -> None:
        reacquire = config.get("reacquire", {})
        thresholds = config.get("thresholds", {})
        identity = config.get("identity", {})
        self.confirm_frames = int(reacquire.get("confirm_frames", 3))
        self.confirm_iou_gate = float(reacquire.get("confirm_iou_gate", 0.3))
        self.confirm_max_gap = int(reacquire.get("confirm_max_gap", 1))
        self.learn_distractors = bool(reacquire.get("learn_distractors", True))
        self.distractor_min_score = float(reacquire.get("distractor_min_score", 0.5))
        self.reacquire_threshold = float(thresholds.get("reacquire_threshold", 0.75))
        self.min_margin = float(identity.get("min_margin", 0.12))

    def reset(self) -> None:
        self.buffer = ConfirmationBuffer(self.confirm_frames, self.confirm_max_gap, self.confirm_iou_gate)

    def _pending(self, candidate: dict | None = None, reid: float = 0.0) -> ReacqOutcome:
        return ReacqOutcome(False, None, reid, candidate, self.buffer.streak, self.confirm_frames)

    def attempt(self, frame: Any, candidates: list[dict]) -> ReacqOutcome:
        if not candidates:
            self.buffer.miss()
            return self._pending()
        best = self.engine.best(frame, candidates)
        if best is None:
            self.buffer.miss()
            return self._pending()

        bbox = tuple(int(v) for v in best["bbox"])
        reid = float(best.get("reid_score", 0.0))
        # Anchor gate: identity must agree with the long-term anchor (drm+wrm),
        # never the drifted short-term ram, and clear the re-acquire threshold.
        anchor = self.memory.anchor_score(frame, bbox)
        on_anchor = anchor.get("negative_margin", 0.0) >= self.min_margin
        strong = reid >= self.reacquire_threshold
        if not (on_anchor and strong):
            # A confident-looking detection that fails the anchor is a look-alike:
            # teach the negative bank so future frames reject it faster.
            if self.learn_distractors and float(best.get("score", 0.0)) >= self.distractor_min_score:
                self.memory.add_distractor(frame, bbox)
            self.buffer.miss()
            return self._pending(best, reid)

        self.buffer.push(bbox, reid)
        return ReacqOutcome(
            confirmed=self.buffer.confirmed(),
            bbox=self.buffer.best_bbox(),
            reid_score=reid,
            candidate=best,
            confirming=self.buffer.streak,
            need=self.confirm_frames,
        )
