from __future__ import annotations

from typing import Iterable


BBox = tuple[int, int, int, int]


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(int(x), max(0, width - 1)))
    y = max(0, min(int(y), max(0, height - 1)))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def bbox_area(bbox: BBox) -> int:
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def nms(boxes: Iterable[BBox], threshold: float = 0.35) -> list[BBox]:
    ordered = sorted(boxes, key=bbox_area, reverse=True)
    kept: list[BBox] = []
    for box in ordered:
        if all(bbox_iou(box, other) < threshold for other in kept):
            kept.append(box)
    return kept
