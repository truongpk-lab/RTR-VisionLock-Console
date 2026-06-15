from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .utils import BBox, clamp_bbox, nms


class OpenCVProposalDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.max_candidates = int(config.get("runtime", {}).get("max_candidates", 20))

    def detect(self, frame: np.ndarray) -> list[dict]:
        if cv2 is None or frame is None:
            return []
        height, width = frame.shape[:2]
        scale = 640.0 / max(width, height)
        work = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 45, 120)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[BBox] = []
        min_area = max(300, int(width * height * 0.002))
        max_area = int(width * height * 0.65)
        inv = 1.0 / scale if scale < 1 else 1.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = clamp_bbox((int(x * inv), int(y * inv), int(w * inv), int(h * inv)), width, height)
            area = box[2] * box[3]
            if min_area <= area <= max_area and box[2] > 12 and box[3] > 12:
                boxes.append(box)

        if not boxes:
            # Last-resort real image proposal: center crop from the actual frame.
            fallback = (width // 3, height // 3, width // 3, height // 3)
            boxes.append(clamp_bbox(fallback, width, height))

        result = []
        for idx, box in enumerate(nms(boxes)[: self.max_candidates]):
            result.append({"id": f"C{idx}", "bbox": list(box), "score": round(0.55 + 0.4 / (idx + 1), 3)})
        return result
