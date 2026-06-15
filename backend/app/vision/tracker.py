from __future__ import annotations

from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .utils import BBox


class OpenCVTracker:
    def __init__(self, config: dict[str, Any], light: bool = False) -> None:
        self.config = config
        # Light trackers favour speed (KCF/MOSSE) so we can run one per candidate
        # in CANDIDATE_TRACKING; the main/learning tracker favours CSRT accuracy.
        self.light = light
        self.tracker = None
        self.kind = "none"
        self.last_bbox: BBox | None = None

    def _factory(self):
        if cv2 is None:
            return None, "none"
        accurate = [
            ("CSRT", lambda: cv2.TrackerCSRT_create()),
            ("KCF", lambda: cv2.TrackerKCF_create()),
            ("MIL", lambda: cv2.TrackerMIL_create()),
        ]
        fast = [
            ("KCF", lambda: cv2.TrackerKCF_create()),
            ("MIL", lambda: cv2.TrackerMIL_create()),
            ("CSRT", lambda: cv2.TrackerCSRT_create()),
        ]
        constructors = fast if self.light else accurate
        legacy = getattr(cv2, "legacy", None)
        if legacy is not None:
            legacy_fast = [
                ("legacy.MOSSE", lambda: legacy.TrackerMOSSE_create()),
                ("legacy.KCF", lambda: legacy.TrackerKCF_create()),
                ("legacy.MIL", lambda: legacy.TrackerMIL_create()),
            ]
            legacy_accurate = [
                ("legacy.CSRT", lambda: legacy.TrackerCSRT_create()),
                ("legacy.KCF", lambda: legacy.TrackerKCF_create()),
                ("legacy.MIL", lambda: legacy.TrackerMIL_create()),
            ]
            constructors = constructors + (legacy_fast if self.light else legacy_accurate)
        for name, build in constructors:
            try:
                return build(), name
            except Exception:
                continue
        return None, "none"

    def init(self, frame, bbox: BBox) -> bool:
        tracker, kind = self._factory()
        if tracker is None:
            return False
        self.tracker = tracker
        self.kind = kind
        self.last_bbox = bbox
        return bool(self.tracker.init(frame, tuple(bbox)))

    def update(self, frame) -> tuple[bool, BBox | None]:
        if self.tracker is None:
            return False, self.last_bbox
        ok, bbox = self.tracker.update(frame)
        if not ok:
            return False, self.last_bbox
        self.last_bbox = tuple(int(v) for v in bbox)
        return True, self.last_bbox
