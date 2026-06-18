"""Confidence-gated 3-tier tracking policy (LTMU-style meta-updater).

This is the brain of the "after-lock" tracking layer. Every frame the session
grades the target into a 0..1 ``score``; this policy turns that score (plus short
history) into *which tier should run next* and *what UI state to show*, with
hysteresis so the system does not flap at the band edges:

    score >= stable_threshold        -> Tier A  NORMAL  (UETrack-B, fast local track)
    uncertain <= score < stable      -> Tier B  REFIND  (EVPTrack, local re-find)
    score < uncertain (for N frames) -> Tier C  LOST    (global YOLO+ReID re-detect)

Tier C handles the "target disappeared for a while and reappeared somewhere far"
case that local trackers (A/B) structurally cannot, because it searches the whole
frame instead of a window around the last position.

Pure arithmetic over plain floats — no cv2/torch — so it is cheap to call every
frame and unit-testable without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.core.states import TrackingState


class TrackMode(str, Enum):
    NORMAL = "NORMAL"  # Tier A: high-confidence local tracker
    REFIND = "REFIND"  # Tier B: low-confidence local re-find


@dataclass
class PolicyDecision:
    mode: TrackMode  # which tier's tracker to run on the NEXT frame
    state: TrackingState  # UI/log state for THIS frame
    seed_refind: bool  # entered REFIND -> (re)seed the re-find tracker on last good box
    reinit_normal: bool  # recovered to NORMAL -> re-init the normal tracker on the recovered box
    reacquire: bool  # escalate to Tier C global re-detect


class TrackingPolicy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.configure(config)
        self.reset()

    def configure(self, config: dict[str, Any]) -> None:
        """(Re)read thresholds from config without disturbing the live counters."""
        thresholds = config.get("thresholds", {})
        tracking = config.get("tracking", {})
        self.stable = float(thresholds.get("stable_threshold", 0.70))
        self.uncertain = float(thresholds.get("uncertain_threshold", 0.45))
        self.lost_frames = int(thresholds.get("lost_frames", 5))
        # Consecutive sub-stable (but not lost) frames tolerated before handing the
        # target from the normal tracker to the re-find tracker.
        self.refind_after = max(1, int(tracking.get("refind_after", 2)))
        # Consecutive sub-min_similarity frames (tracker still ok, but on the wrong
        # object) inside UNCERTAIN before forcing LOST -> global re-detect. Larger
        # than lost_frames: this is a soft "tracking-something-wrong" loss, so we
        # are more patient than for a hard ok=False loss.
        self.identity_lost_frames = max(1, int(thresholds.get("identity_lost_frames", 12)))

    def reset(self) -> None:
        """Called on every fresh lock / successful re-acquire."""
        self.mode = TrackMode.NORMAL
        self.lost_count = 0  # frames clearly below the uncertain band
        self.soft_count = 0  # consecutive frames inside the uncertain band
        self.identity_lost_count = 0  # consecutive ok frames with low identity

    # Read alias for intent at call sites.
    def on_lock(self) -> None:
        self.reset()

    def update(self, score: float, ok: bool, identity_lost: bool = False) -> PolicyDecision:
        # Track how long the tracker has been ok but on a low-identity (wrong)
        # box. Any solid/recovered frame clears the streak.
        if ok and identity_lost:
            self.identity_lost_count += 1
        else:
            self.identity_lost_count = 0

        # Tier A: solid frame -> normal tracking.
        if ok and score >= self.stable:
            reinit_normal = self.mode == TrackMode.REFIND
            self.mode = TrackMode.NORMAL
            self.lost_count = 0
            self.soft_count = 0
            return PolicyDecision(self.mode, TrackingState.STABLE, False, reinit_normal, False)

        # Tier B: drifting but still plausibly on target -> local re-find.
        if ok and score >= self.uncertain:
            self.soft_count += 1
            self.lost_count = 0
            seed_refind = False
            if self.mode == TrackMode.NORMAL and self.soft_count >= self.refind_after:
                self.mode = TrackMode.REFIND
                seed_refind = True
            # Stuck-on-wrong-object escape: a tracker that stays ok inside the
            # UNCERTAIN band with persistently low identity never reaches the LOST
            # branch below (lost_count only counts ok=False frames). Force the
            # escalation to global re-detect so the true target can be re-found.
            if self.identity_lost_count >= self.identity_lost_frames:
                if self.mode == TrackMode.NORMAL:
                    self.mode = TrackMode.REFIND
                    seed_refind = True
                return PolicyDecision(self.mode, TrackingState.LOST, seed_refind, False, True)
            return PolicyDecision(self.mode, TrackingState.UNCERTAIN, seed_refind, False, False)

        # Tier C path: lost this frame. Slip straight to the re-find tracker if we
        # were still tracking normally, and escalate to global re-detect once the
        # target has been gone for `lost_frames` frames.
        self.lost_count += 1
        seed_refind = False
        if self.mode == TrackMode.NORMAL:
            self.mode = TrackMode.REFIND
            seed_refind = True
        if self.lost_count >= self.lost_frames:
            return PolicyDecision(self.mode, TrackingState.LOST, seed_refind, False, True)
        return PolicyDecision(self.mode, TrackingState.UNCERTAIN, seed_refind, False, False)


@dataclass
class GateDecision:
    """What the session is allowed to do this frame, derived from the policy.

    The session never branches on the raw ``TrackingState`` for gating; it reads
    these explicit flags instead, so "turn detection off after lock" and "freeze
    memory while uncertain" live in exactly one place.
    """

    policy: PolicyDecision
    allow_memory_update: bool  # LOCKED (STABLE): safe to write the appearance bank
    freeze_memory: bool  # UNCERTAIN: hold the bank steady, do not learn
    run_detection: bool  # LOST: the ONLY state that may run the global detector
    confidence_state: str  # "LOCKED" | "UNCERTAIN" | "LOST" (for logs/snapshot)


# Coarse confidence labels surfaced to logs/UI, mapped from the policy state.
_CONFIDENCE_LABEL = {
    TrackingState.STABLE: "LOCKED",
    TrackingState.UNCERTAIN: "UNCERTAIN",
    TrackingState.LOST: "LOST",
}


class ConfidenceManager:
    """Owns the confidence state machine and the per-frame gating decision.

    Thin wrapper over :class:`TrackingPolicy`: it does not re-implement the
    transition logic (that stays tested in ``test_policy.py``); it only translates
    the policy's decision into the gate flags the session needs. Pure logic —
    no cv2/torch.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.policy = TrackingPolicy(config)
        self.confidence_state = "LOCKED"

    def configure(self, config: dict[str, Any]) -> None:
        self.policy.configure(config)

    def reset(self) -> None:
        self.policy.reset()
        self.confidence_state = "LOCKED"

    def on_lock(self) -> None:
        self.policy.on_lock()
        self.confidence_state = "LOCKED"

    @property
    def mode(self) -> TrackMode:
        return self.policy.mode

    def update(self, score: float, ok: bool, identity_lost: bool = False) -> GateDecision:
        decision = self.policy.update(score, ok, identity_lost)
        label = _CONFIDENCE_LABEL.get(decision.state, "UNCERTAIN")
        self.confidence_state = label
        return GateDecision(
            policy=decision,
            allow_memory_update=decision.state == TrackingState.STABLE,
            freeze_memory=decision.state == TrackingState.UNCERTAIN,
            run_detection=decision.reacquire,
            confidence_state=label,
        )
