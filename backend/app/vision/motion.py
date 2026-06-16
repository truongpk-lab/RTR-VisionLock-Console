"""SAMURAI-style motion-aware decision logic.

This module isolates the two core contributions of SAMURAI
(https://github.com/yangchris11/samurai) as small, dependency-free helpers so
they can be reused by both the SAM2 video path and the OpenCV fallback, and unit
tested without torch/cv2:

1. ``select_motion_aware_mask`` — when SAM2 emits several candidate masks, pick
   the one that best agrees with the Kalman motion prediction, not just the one
   with the highest raw affinity. This is what keeps the lock on the right
   instance when a similar object passes nearby.

2. ``memory_admission`` — decide whether the current frame is clean enough to be
   written into the appearance memory bank. Frames that look ambiguous (low
   affinity, weak motion, or too close to a distractor) are rejected so the
   memory never drifts toward a look-alike.

Everything here is pure Python/numpy-free arithmetic over plain bboxes, so it
carries no heavy runtime cost on Jetson and is safe to call every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .utils import BBox, bbox_iou


@dataclass
class MaskCandidate:
    """One SAM2 mask hypothesis reduced to a bbox plus its model affinity."""

    bbox: BBox
    affinity: float


@dataclass
class MaskSelection:
    index: int
    bbox: BBox
    affinity: float
    motion_iou: float
    distractor_penalty: float
    score: float


@dataclass
class MotionWeights:
    """Weights for the SAMURAI mask-selection score.

    ``alpha_kf`` is the weight given to the Kalman motion agreement (KF-IoU); the
    remaining ``1 - alpha_kf`` weights the raw mask affinity. ``distractor`` is a
    straight penalty for overlapping a known negative box.
    """

    alpha_kf: float = 0.25
    distractor: float = 0.25

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MotionWeights":
        samurai = config.get("samurai", {}) if isinstance(config, dict) else {}
        return cls(
            alpha_kf=float(samurai.get("alpha_kf", 0.25)),
            distractor=float(samurai.get("distractor_penalty", 0.25)),
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def select_motion_aware_mask(
    candidates: Sequence[MaskCandidate],
    predicted_bbox: BBox | None,
    negatives: Sequence[BBox] | None = None,
    weights: MotionWeights | None = None,
) -> MaskSelection | None:
    """Pick the mask that best balances appearance affinity and motion.

    ``score = (1 - alpha_kf) * affinity + alpha_kf * KF_IoU - distractor * penalty``

    When there is no Kalman prediction yet (first lock), the motion term falls
    back to the affinity so behaviour is unchanged before motion is established.
    """
    weights = weights or MotionWeights()
    negatives = list(negatives or [])
    best: MaskSelection | None = None
    for idx, candidate in enumerate(candidates):
        if candidate.bbox is None:
            continue
        affinity = _clamp01(candidate.affinity)
        if predicted_bbox is not None:
            motion_iou = bbox_iou(predicted_bbox, candidate.bbox)
        else:
            motion_iou = affinity
        distractor_penalty = max(
            (bbox_iou(candidate.bbox, negative) for negative in negatives),
            default=0.0,
        )
        score = (
            (1.0 - weights.alpha_kf) * affinity
            + weights.alpha_kf * motion_iou
            - weights.distractor * distractor_penalty
        )
        if best is None or score > best.score:
            best = MaskSelection(
                index=idx,
                bbox=candidate.bbox,
                affinity=affinity,
                motion_iou=round(motion_iou, 4),
                distractor_penalty=round(distractor_penalty, 4),
                score=round(score, 4),
            )
    return best


@dataclass
class AdmissionThresholds:
    """Gates for SAMURAI motion-aware memory selection."""

    min_affinity: float = 0.5
    min_positive: float = 0.6
    max_negative: float = 0.5
    min_motion: float = 0.5
    min_margin: float = 0.10

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AdmissionThresholds":
        samurai = config.get("samurai", {}) if isinstance(config, dict) else {}
        adm = samurai.get("memory_admission", {}) if isinstance(samurai, dict) else {}
        return cls(
            min_affinity=float(adm.get("min_affinity", 0.5)),
            min_positive=float(adm.get("min_positive", 0.6)),
            max_negative=float(adm.get("max_negative", 0.5)),
            min_motion=float(adm.get("min_motion", 0.5)),
            min_margin=float(adm.get("min_margin", 0.10)),
        )


@dataclass
class AdmissionDecision:
    admit: bool
    reason: str


def memory_admission(
    positive_sim: float,
    negative_sim: float,
    motion_consistency: float,
    affinity: float,
    thresholds: AdmissionThresholds | None = None,
) -> AdmissionDecision:
    """Return whether this frame may be written into the appearance memory.

    A frame is admitted only when it is unambiguous on every axis: the mask/track
    is confident, it looks like the target, it does NOT look like a nearby
    distractor, and the motion is consistent. Rejecting otherwise is what stops
    the memory bank from slowly absorbing a similar object during a crossing.
    """
    thresholds = thresholds or AdmissionThresholds()
    margin = float(positive_sim) - float(negative_sim)
    if affinity < thresholds.min_affinity:
        return AdmissionDecision(False, "low_affinity")
    if positive_sim < thresholds.min_positive:
        return AdmissionDecision(False, "low_positive")
    if negative_sim > thresholds.max_negative:
        return AdmissionDecision(False, "near_distractor")
    if motion_consistency < thresholds.min_motion:
        return AdmissionDecision(False, "unstable_motion")
    if margin < thresholds.min_margin:
        return AdmissionDecision(False, "low_margin")
    return AdmissionDecision(True, "ok")
