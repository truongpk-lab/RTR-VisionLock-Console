from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def track_score(
    tracker_confidence: float,
    appearance_similarity: float,
    motion_consistency: float,
    mask_quality: float = 1.0,
) -> float:
    return clamp(
        0.35 * tracker_confidence
        + 0.25 * appearance_similarity
        + 0.20 * motion_consistency
        + 0.20 * mask_quality
    )


@dataclass
class MetricState:
    fps: float = 0.0
    latency_ms: float = 0.0
    gpu: str = "N/A"
    camera: str = "OFFLINE"
    track_score: float = 0.0
    confidence: float = 0.0
    similarity: float = 0.0
    mask_iou: float = 0.0
    kalman_error: float = 0.0
    motion: str = "IDLE"
    candidates: int = 0

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "gpu": self.gpu,
            "camera": self.camera,
            "track_score": round(self.track_score, 3),
            "confidence": round(self.confidence, 3),
            "similarity": round(self.similarity, 3),
            "mask_iou": round(self.mask_iou, 3),
            "kalman_error": round(self.kalman_error, 2),
            "motion": self.motion,
            "candidates": self.candidates,
        }
