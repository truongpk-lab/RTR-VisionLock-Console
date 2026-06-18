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
from .utils import BBox, bbox_center, bbox_iou


class ConfirmationBuffer:
    """Counts consecutive, consistent confirmations of a candidate.

    ``need`` consecutive consistent hits confirm a re-lock. A short dropout is
    tolerated up to ``max_gap`` misses; an inconsistent hit (a different location)
    resets the streak so a flickering false match can never accumulate.

    Consistency is an OR of three gates so a fast-moving target is not dropped just
    because consecutive boxes barely overlap:
      * IoU overlap >= ``iou_gate`` (spatial), OR
      * centres within ``center_gate`` * box size (handles fast motion, IoU~0), OR
      * appearance ``reid_score`` >= ``identity_gate`` (same target, moved far).
    The latter two default to OFF (0.0 / 1.01) so the plain IoU behaviour is the
    default; the session enables them from config.
    """

    def __init__(
        self,
        need: int,
        max_gap: int,
        iou_gate: float,
        center_gate: float = 0.0,
        identity_gate: float = 1.01,
    ) -> None:
        self.need = max(1, int(need))
        self.max_gap = max(0, int(max_gap))
        self.iou_gate = float(iou_gate)
        self.center_gate = float(center_gate)
        self.identity_gate = float(identity_gate)
        self.reset()

    def reset(self) -> None:
        self.streak = 0
        self.gap = 0
        self.last_bbox: BBox | None = None
        self._best_bbox: BBox | None = None
        self._best_score = -1.0

    def _consistent(self, prev: BBox, cur: BBox, reid_score: float) -> bool:
        if bbox_iou(prev, cur) >= self.iou_gate:
            return True
        if self.center_gate > 0.0:
            px, py = bbox_center(prev)
            cx, cy = bbox_center(cur)
            distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if distance <= self.center_gate * max(cur[2], cur[3], 1):
                return True
        return reid_score >= self.identity_gate

    def push(self, bbox: BBox, reid_score: float) -> None:
        bbox = tuple(int(v) for v in bbox)
        consistent = self.last_bbox is None or self._consistent(self.last_bbox, bbox, reid_score)
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

    def best_score(self) -> float:
        return self._best_score


class MultiHypothesisBuffer:
    """Up to ``k`` independent confirmation streaks running in parallel.

    With a single buffer, a real but second-best candidate has its streak wiped
    every frame the detector's top pick lands elsewhere. Keeping k streaks lets a
    consistent runner-up accumulate; we re-lock as soon as ANY hypothesis confirms.
    With ``k == 1`` this is exactly the single-buffer behaviour.
    """

    def __init__(self, k: int, *buffer_args: Any) -> None:
        self.k = max(1, int(k))
        self._buffer_args = buffer_args
        self.reset()

    def reset(self) -> None:
        self._hyps: list[ConfirmationBuffer] = []

    def _match(self, bbox: BBox, reid_score: float) -> ConfirmationBuffer | None:
        for buf in self._hyps:
            if buf.last_bbox is not None and buf._consistent(buf.last_bbox, bbox, reid_score):
                return buf
        return None

    def update(self, observations: list[tuple[BBox, float]]) -> None:
        """Advance every hypothesis from this frame's gate-passing observations."""
        matched: set[int] = set()
        for bbox, reid in observations:
            buf = self._match(bbox, reid)
            if buf is None:
                if len(self._hyps) >= self.k:
                    buf = min(self._hyps, key=lambda b: b.streak)  # evict the weakest
                    buf.reset()
                else:
                    buf = ConfirmationBuffer(*self._buffer_args)
                    self._hyps.append(buf)
            buf.push(bbox, reid)
            matched.add(id(buf))
        for buf in self._hyps:
            if id(buf) not in matched:
                buf.miss()

    @property
    def streak(self) -> int:
        return max((b.streak for b in self._hyps), default=0)

    def confirmed_buffer(self) -> ConfirmationBuffer | None:
        for buf in self._hyps:
            if buf.confirmed():
                return buf
        return None


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
        self.reset()

    def configure(self, config: dict[str, Any]) -> None:
        reacquire = config.get("reacquire", {})
        thresholds = config.get("thresholds", {})
        identity = config.get("identity", {})
        self.confirm_frames = int(reacquire.get("confirm_frames", 3))
        self.confirm_iou_gate = float(reacquire.get("confirm_iou_gate", 0.3))
        self.confirm_max_gap = int(reacquire.get("confirm_max_gap", 1))
        # OR-gates so a fast/far re-appearance is not dropped on low IoU. Default
        # off (0.0 / 1.01) -> plain IoU behaviour unless configured.
        self.confirm_center_gate = float(reacquire.get("confirm_center_gate_scale", 0.0))
        self.confirm_identity_gate = float(reacquire.get("confirm_identity_gate", 1.01))
        # Parallel confirmation hypotheses. 1 == single-buffer (default, unchanged).
        self.top_k = max(1, int(reacquire.get("top_k", 1)))
        self.learn_distractors = bool(reacquire.get("learn_distractors", True))
        self.distractor_min_score = float(reacquire.get("distractor_min_score", 0.5))
        self.reacquire_threshold = float(thresholds.get("reacquire_threshold", 0.75))
        self.min_margin = float(identity.get("min_margin", 0.12))
        # Instant re-lock fast-path (default OFF: 1.01 is unreachable). See _fast_relock.
        self.fast_relock_identity = float(reacquire.get("fast_relock_identity", 1.01))
        self.fast_relock_max_age = float(reacquire.get("fast_relock_max_age", 2.0))

    def reset(self) -> None:
        args = (
            self.confirm_frames,
            self.confirm_max_gap,
            self.confirm_iou_gate,
            self.confirm_center_gate,
            self.confirm_identity_gate,
        )
        self.buffer = ConfirmationBuffer(*args)
        self.multi = MultiHypothesisBuffer(self.top_k, *args)

    @property
    def confirming(self) -> int:
        """Current best confirmation streak (across hypotheses when top_k > 1)."""
        return self.multi.streak if self.top_k > 1 else self.buffer.streak

    def _pending(self, candidate: dict | None = None, reid: float = 0.0) -> ReacqOutcome:
        return ReacqOutcome(False, None, reid, candidate, self.buffer.streak, self.confirm_frames)

    def _gate(self, frame: Any, candidate: dict) -> tuple[BBox, float, bool]:
        """Anchor + strength gate; teach the negative bank on a confident miss."""
        bbox = tuple(int(v) for v in candidate["bbox"])
        reid = float(candidate.get("reid_score", 0.0))
        # Anchor gate: identity must agree with the long-term anchor (drm+wrm),
        # never the drifted short-term ram, and clear the re-acquire threshold.
        anchor = self.memory.anchor_score(frame, bbox)
        on_anchor = anchor.get("negative_margin", 0.0) >= self.min_margin
        strong = reid >= self.reacquire_threshold
        passed = on_anchor and strong
        if not passed and self.learn_distractors and float(candidate.get("score", 0.0)) >= self.distractor_min_score:
            self.memory.add_distractor(frame, bbox)
        return bbox, reid, passed

    def _fast_relock(self, reid: float, lost_age_sec: float) -> bool:
        """Instant re-lock when the appearance match is unambiguous and recent.

        Only reached for a candidate that already passed the anchor+strength gate,
        so this is a high-confidence shortcut past the multi-frame confirmation. It
        is bounded to the first ``fast_relock_max_age`` seconds of loss, where the
        re-acquire engine still weights motion, so the pick is corroborated by
        position too -- a far look-alike after a long loss still has to confirm over
        the streak. Default OFF (``fast_relock_identity`` 1.01 is unreachable).
        """
        if reid < self.fast_relock_identity:
            return False
        return self.fast_relock_max_age <= 0.0 or lost_age_sec <= self.fast_relock_max_age

    def attempt(self, frame: Any, candidates: list[dict], lost_age_sec: float = 0.0) -> ReacqOutcome:
        if self.top_k > 1:
            return self._attempt_multi(frame, candidates, lost_age_sec)
        if not candidates:
            self.buffer.miss()
            return self._pending()
        best = self.engine.best(frame, candidates, lost_age_sec)
        if best is None:
            self.buffer.miss()
            return self._pending()
        bbox, reid, passed = self._gate(frame, best)
        if not passed:
            self.buffer.miss()
            return self._pending(best, reid)
        if self._fast_relock(reid, lost_age_sec):
            return ReacqOutcome(True, bbox, reid, best, self.confirm_frames, self.confirm_frames)
        self.buffer.push(bbox, reid)
        return ReacqOutcome(
            confirmed=self.buffer.confirmed(),
            bbox=self.buffer.best_bbox(),
            reid_score=reid,
            candidate=best,
            confirming=self.buffer.streak,
            need=self.confirm_frames,
        )

    def _attempt_multi(self, frame: Any, candidates: list[dict], lost_age_sec: float) -> ReacqOutcome:
        """Top-K path: advance up to ``top_k`` parallel hypotheses, confirm any."""
        if not candidates:
            self.multi.update([])
            return ReacqOutcome(False, None, 0.0, None, self.multi.streak, self.confirm_frames)
        ranked = self.engine.rank(frame, candidates, lost_age_sec)[: self.top_k]
        observations: list[tuple[BBox, float]] = []
        best_candidate: dict | None = None
        best_reid = 0.0
        best_bbox: BBox | None = None
        for candidate in ranked:
            bbox, reid, passed = self._gate(frame, candidate)
            if passed:
                observations.append((bbox, reid))
                if best_candidate is None:  # ranked desc -> first passer is the strongest
                    best_candidate, best_reid, best_bbox = candidate, reid, bbox
        # Fast-path: the top-ranked passer (motion-corroborated by the engine) is an
        # unambiguous, recent appearance match -> re-lock now without the streak.
        if best_candidate is not None and self._fast_relock(best_reid, lost_age_sec):
            return ReacqOutcome(True, best_bbox, best_reid, best_candidate, self.confirm_frames, self.confirm_frames)
        self.multi.update(observations)
        confirmed = self.multi.confirmed_buffer()
        if confirmed is not None:
            return ReacqOutcome(True, confirmed.best_bbox(), confirmed.best_score(), best_candidate, confirmed.streak, self.confirm_frames)
        return ReacqOutcome(False, None, best_reid, best_candidate, self.multi.streak, self.confirm_frames)
