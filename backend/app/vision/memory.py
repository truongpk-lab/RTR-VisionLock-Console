from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .encoders import build_encoder
from .motion import AdmissionDecision, AdmissionThresholds, memory_admission
from .utils import BBox


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


class MemoryBank:
    """Appearance memory with positive (RAM/DRM) and negative banks.

    The encoder is pluggable (see ``encoders.build_encoder``) and writes into the
    positive bank are gated by a SAMURAI-style motion-aware admission test so a
    nearby look-alike cannot poison the lock.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        memory_cfg = config.get("memory", {})
        identity_cfg = config.get("identity", {})
        self.ram_slots = int(memory_cfg.get("ram_slots", 8))
        self.wrm_slots = int(memory_cfg.get("wrm_slots", self.ram_slots))
        self.drm_slots = int(memory_cfg.get("drm_slots", 8))
        self.negative_slots = int(identity_cfg.get("negative_slots", memory_cfg.get("negative_slots", 8)))
        # Working/long-term consolidation pacing (off the SAMURAI-admitted RAM stream).
        self.working_promote_every = max(1, int(memory_cfg.get("working_promote_every", 12)))
        self.long_term_min_margin = float(memory_cfg.get("long_term_min_margin", 0.25))
        self._admit_since_promote = 0
        self.distractors_added = 0
        self.base_id = f"{memory_cfg.get('base_id_prefix', 'TGT')}-8842-A"
        self.min_margin = float(identity_cfg.get("min_margin", 0.12))
        self.last_margin = 0.0

        self.encoder = build_encoder(config)
        self.identity_backend = self.encoder.name
        self.feature_dim = int(self.encoder.feature_dim or memory_cfg.get("feature_dim", 1026))
        self.admission = AdmissionThresholds.from_config(config)
        self.last_admission = "init"
        self.admitted_count = 0
        self.rejected_count = 0

        # Three positive tiers (SAMURAI RAM/DRM extended with a working tier):
        #   ram = short-term  (fast, every admitted STABLE frame)
        #   wrm = working-term (medium, periodic promotion from ram)
        #   drm = long-term    (stable identity ANCHOR; seeded at lock, rarely updated)
        # Global Re-ID matches the anchor (drm+wrm), never the volatile ram.
        self.ram: deque[np.ndarray] = deque(maxlen=self.ram_slots)
        self.wrm: deque[np.ndarray] = deque(maxlen=self.wrm_slots)
        self.drm: deque[np.ndarray] = deque(maxlen=self.drm_slots)
        self.negative: deque[np.ndarray] = deque(maxlen=self.negative_slots)

    def configure(self, config: dict[str, Any]) -> None:
        """Apply config-tunable thresholds and bank sizes live (Memory Config UI).

        Threshold/margin changes take effect immediately. Slot-count changes resize
        the banks in place WITHOUT dropping already-learned features, so it is safe
        even while a target is locked. The encoder itself is not swapped here (that
        would invalidate stored features) — an encoder change applies on next lock.
        """
        memory_cfg = config.get("memory", {})
        identity_cfg = config.get("identity", {})
        self.min_margin = float(identity_cfg.get("min_margin", self.min_margin))
        self.long_term_min_margin = float(memory_cfg.get("long_term_min_margin", self.long_term_min_margin))
        self.working_promote_every = max(1, int(memory_cfg.get("working_promote_every", self.working_promote_every)))
        self.admission = AdmissionThresholds.from_config(config)
        self.ram_slots = int(memory_cfg.get("ram_slots", self.ram_slots))
        self.wrm_slots = int(memory_cfg.get("wrm_slots", self.wrm_slots))
        self.drm_slots = int(memory_cfg.get("drm_slots", self.drm_slots))
        self.negative_slots = int(identity_cfg.get("negative_slots", memory_cfg.get("negative_slots", self.negative_slots)))
        self.ram = deque(self.ram, maxlen=self.ram_slots)
        self.wrm = deque(self.wrm, maxlen=self.wrm_slots)
        self.drm = deque(self.drm, maxlen=self.drm_slots)
        self.negative = deque(self.negative, maxlen=self.negative_slots)

    def extract(self, frame: np.ndarray, bbox: BBox, mask: np.ndarray | None = None) -> np.ndarray | None:
        return self.encoder.extract(frame, bbox, mask)

    def initialize(self, frame: np.ndarray, bbox: BBox) -> bool:
        feature = self.extract(frame, bbox)
        if feature is None:
            return False
        self.ram.clear()
        self.wrm.clear()
        self.drm.clear()
        self.ram.append(feature)
        self.drm.append(feature)
        return True

    def load_samples(self, samples: list[np.ndarray], negative_samples: list[np.ndarray] | None = None) -> bool:
        """Seed RAM/DRM from features collected during LEARNING_TARGET."""
        clean = [s for s in samples if s is not None]
        if not clean:
            return False
        self.ram.clear()
        self.wrm.clear()
        self.drm.clear()
        self._admit_since_promote = 0
        # Newest samples are the most representative of the locked appearance.
        for feature in clean[-self.ram_slots :]:
            self.ram.append(feature)
        # Spread earlier samples into the working + long-term anchor tiers so the
        # re-detection anchor (drm+wrm) is populated immediately at lock.
        spread = clean[:: max(1, len(clean) // self.drm_slots or 1)]
        for feature in spread[: self.drm_slots]:
            self.drm.append(feature)
        for feature in spread[: self.wrm_slots]:
            self.wrm.append(feature)
        if negative_samples is not None:
            self.negative.clear()
            for feature in [s for s in negative_samples if s is not None][-self.negative_slots :]:
                self.negative.append(feature)
        return True

    def update_ram(self, frame: np.ndarray, bbox: BBox) -> None:
        feature = self.extract(frame, bbox)
        if feature is not None:
            self.ram.append(feature)

    def update_drm(self, frame: np.ndarray, bbox: BBox) -> None:
        feature = self.extract(frame, bbox)
        if feature is not None:
            self.drm.append(feature)

    def update_negative(self, frame: np.ndarray, bbox: BBox) -> None:
        feature = self.extract(frame, bbox)
        if feature is not None:
            self.negative.append(feature)

    def consider_update(
        self,
        frame: np.ndarray,
        bbox: BBox,
        motion_consistency: float,
        affinity: float,
        scores: dict[str, float] | None = None,
    ) -> AdmissionDecision:
        """SAMURAI motion-aware memory selection.

        Score the current crop, then admit it into RAM only if it is confident,
        on-identity, far from any distractor, and moving consistently. Returns the
        decision so callers can surface why a frame was skipped. ``scores`` may be
        passed in to reuse an identity score already computed this frame.
        """
        if scores is None:
            scores = self.score(frame, bbox)
        decision = memory_admission(
            positive_sim=scores["positive_similarity"],
            negative_sim=scores["negative_similarity"],
            motion_consistency=float(motion_consistency),
            affinity=float(affinity),
            thresholds=self.admission,
        )
        self.last_admission = decision.reason
        if decision.admit:
            self.update_ram(frame, bbox)
            self.admitted_count += 1
        else:
            self.rejected_count += 1
        return decision

    @staticmethod
    def _zero_scores() -> dict[str, float]:
        return {
            "positive_similarity": 0.0,
            "negative_similarity": 0.0,
            "identity_score": 0.0,
            "negative_margin": 0.0,
        }

    def _identity(self, feature: np.ndarray, positives: list[np.ndarray]) -> dict[str, float]:
        positive = max((_cosine(feature, item) for item in positives), default=0.0)
        negative = max((_cosine(feature, item) for item in self.negative), default=0.0)
        margin = positive - negative
        # Keep old callers on a 0..1 confidence-like score while making nearby
        # distractors costly. With no negatives this stays close to positive.
        identity = max(0.0, min(1.0, positive - 0.45 * negative + 0.20))
        if self.negative and margin < self.min_margin:
            identity *= 0.75
        self.last_margin = margin
        return {
            "positive_similarity": round(positive, 3),
            "negative_similarity": round(negative, 3),
            "identity_score": round(identity, 3),
            "negative_margin": round(margin, 3),
        }

    def score(self, frame: np.ndarray, bbox: BBox) -> dict[str, float]:
        """Live identity score vs short + long-term memory (ram + drm)."""
        if not self.ram:
            return self._zero_scores()
        feature = self.extract(frame, bbox)
        if feature is None:
            return self._zero_scores()
        return self._identity(feature, list(self.ram) + list(self.drm))

    def anchor_score(self, frame: np.ndarray, bbox: BBox) -> dict[str, float]:
        """Re-detection identity score vs the stable ANCHOR only (drm + wrm).

        Global re-detection uses this instead of ``score`` so a drifted/poisoned
        short-term ram can never approve a wrong re-lock — the anchor is the
        curated long-term identity.
        """
        anchor = list(self.drm) + list(self.wrm)
        if not anchor:
            return self._zero_scores()
        feature = self.extract(frame, bbox)
        if feature is None:
            return self._zero_scores()
        return self._identity(feature, anchor)

    def similarity(self, frame: np.ndarray, bbox: BBox) -> float:
        return self.score(frame, bbox)["identity_score"]

    def promote_to_working(self) -> None:
        """Copy the freshest short-term sample into the working tier."""
        if self.ram:
            self.wrm.append(self.ram[-1])

    def consolidate_long_term(self, margin: float) -> bool:
        """Admit the freshest sample into the long-term anchor when on-identity."""
        if self.ram and margin >= self.long_term_min_margin:
            self.drm.append(self.ram[-1])
            return True
        return False

    def consolidate(self, margin: float) -> None:
        """Paced working/long-term update, called after a STABLE ram admit.

        Promotes to the working tier every ``working_promote_every`` admits, and
        only then consolidates into the long-term anchor if the identity margin is
        high — so the anchor drifts slowly and stays trustworthy for re-detection.
        """
        if not self.ram:
            return
        self._admit_since_promote += 1
        if self._admit_since_promote < self.working_promote_every:
            return
        self._admit_since_promote = 0
        self.promote_to_working()
        self.consolidate_long_term(margin)

    def add_distractor(self, frame: np.ndarray, bbox: BBox) -> None:
        """Grow the negative bank with a look-alike rejected during re-detection."""
        feature = self.extract(frame, bbox)
        if feature is not None:
            self.negative.append(feature)
            self.distractors_added += 1

    def to_dict(self) -> dict:
        return {
            "base_id": self.base_id,
            "feature_dim": self.feature_dim,
            "ram_slots": len(self.ram),
            "ram_capacity": self.ram_slots,
            "working_slots": len(self.wrm),
            "working_capacity": self.wrm_slots,
            "drm_slots": len(self.drm),
            "drm_capacity": self.drm_slots,
            "anchor_slots": len(self.drm) + len(self.wrm),
            "positive_slots": len(self.ram) + len(self.wrm) + len(self.drm),
            "negative_slots": len(self.negative),
            "negative_capacity": self.negative_slots,
            "distractors_added": self.distractors_added,
            "identity_backend": self.identity_backend,
            "identity_margin": round(self.last_margin, 3),
            "last_admission": self.last_admission,
            "admitted": self.admitted_count,
            "rejected": self.rejected_count,
            "ram_enabled": True,
            "drm_enabled": True,
        }
