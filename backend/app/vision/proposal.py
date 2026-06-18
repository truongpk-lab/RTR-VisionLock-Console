from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .utils import BBox, clamp_bbox, nms

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional Jetson deployment dependency
    YOLO = None


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


class OpenCVProposalDetector:
    backend = "opencv"

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
            track_id = f"C{idx}"
            result.append(
                {
                    "id": track_id,
                    "track_id": track_id,
                    "bbox": list(box),
                    "score": round(0.55 + 0.4 / (idx + 1), 3),
                    "class_id": None,
                    "class_name": "object",
                    "source": "opencv",
                    "refined": False,
                    "mask_quality": 0.0,
                    "identity_score": 0.0,
                    "negative_margin": 0.0,
                    "motion_score": 0.0,
                    "is_distractor": False,
                    "mask_polygon": None,
                }
            )
        return result

    def to_dict(self) -> dict:
        return {"backend": self.backend, "model_ready": False}


class YoloProposalDetector:
    """YOLO proposal detector with a no-surprises OpenCV fallback.

    Ultralytics can load TensorRT ``.engine`` exports on Jetson. The dependency
    and model file are optional here so the desktop app and tests remain usable
    before deployment assets are installed.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.fallback = OpenCVProposalDetector(config)
        model_cfg = config.get("models", {}).get("proposal", {})
        runtime = config.get("runtime", {})
        self.kind = str(model_cfg.get("type", "yolo11_trt"))
        self.input_size = int(model_cfg.get("input_size", 640))
        self.conf = float(model_cfg.get("conf", 0.35))
        self.iou = float(model_cfg.get("iou", 0.50))
        self.max_candidates = int(runtime.get("max_candidates", 20))
        self.device = str(runtime.get("device", "cuda"))
        self.class_names = list(model_cfg.get("class_names") or [])
        self.model = None
        self.model_path: Path | None = None
        self.backend = "opencv"
        self.last_error = ""

        raw_path = str(model_cfg.get("path", "models/yolo11n_custom.engine"))
        path = Path(raw_path)
        self.model_path = path if path.is_absolute() else _backend_root() / path
        if model_cfg.get("enabled") is True and YOLO is not None and self.model_path.exists():
            try:
                self.model = YOLO(str(self.model_path))
                self.backend = "yolo11_trt" if self.model_path.suffix == ".engine" else "yolo"
            except Exception as exc:  # pragma: no cover - depends on deployment runtime
                self.model = None
                self.last_error = str(exc)
        elif model_cfg.get("enabled") is True:
            if YOLO is None:
                self.last_error = "ultralytics package is not installed"
            elif not self.model_path.exists():
                self.last_error = f"model not found: {self.model_path}"

    @property
    def ready_model(self) -> bool:
        return self.model is not None

    def warmup(self, frame_shape: tuple[int, int]) -> None:
        """Run one dummy detect so the YOLO/TensorRT engine builds before first use."""
        if getattr(self, "_warmed", False) or self.model is None:
            self._warmed = True
            return
        self._warmed = True
        try:  # pragma: no cover - depends on deployment runtime
            h, w = int(frame_shape[0]), int(frame_shape[1])
            self.detect(np.zeros((max(16, h), max(16, w), 3), dtype=np.uint8))
        except Exception:
            pass

    def detect(self, frame: np.ndarray) -> list[dict]:
        if self.model is None:
            return self.fallback.detect(frame)
        try:
            results = self.model.predict(
                source=frame,
                imgsz=self.input_size,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_candidates,
                verbose=False,
                device=self.device if self.device else None,
            )
            if not results:
                return []
            result = results[0]
            names = getattr(result, "names", None) or {}
            detections = []
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                return []
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
            for box, score, class_id in zip(xyxy, confs, classes):
                detections.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score), int(class_id)))
            # Segmentation models (e.g. yolo11n-seg) also return per-instance masks.
            # Surface them as polygons (original image pixels) so the UI can colour
            # the object; detection-only models leave masks None.
            masks = getattr(result, "masks", None)
            polygons = getattr(masks, "xy", None) if masks is not None else None
            return self._build_candidates(detections, frame.shape[1], frame.shape[0], names, polygons)
        except Exception as exc:  # pragma: no cover - depends on deployment runtime
            self.last_error = str(exc)
            return self.fallback.detect(frame)

    def _class_name(self, class_id: int | None, names: dict | list | None = None) -> str:
        if class_id is None:
            return "object"
        if isinstance(names, dict) and class_id in names:
            return str(names[class_id])
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return f"class_{class_id}"

    @staticmethod
    def _simplify_polygon(poly: Any, max_points: int = 60) -> list[list[int]] | None:
        """Down-sample a SAM/YOLO mask polygon to a small int point list for the UI."""
        if poly is None:
            return None
        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 3:
            return None
        if cv2 is not None and len(pts) > 6:
            contour = pts.reshape(-1, 1, 2)
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            pts = approx.reshape(-1, 2)
        if len(pts) > max_points:
            step = int(np.ceil(len(pts) / max_points))
            pts = pts[::step]
        return [[int(round(x)), int(round(y))] for x, y in pts]

    def _build_candidates(
        self,
        detections: list[tuple[float, float, float, float, float, int | None]],
        width: int,
        height: int,
        names: dict | list | None = None,
        polygons: Any = None,
    ) -> list[dict]:
        candidates = []
        for idx, (x1, y1, x2, y2, score, class_id) in enumerate(detections[: self.max_candidates]):
            bbox = clamp_bbox((int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))), width, height)
            track_id = f"Y{idx}"
            mask_polygon = None
            if polygons is not None and idx < len(polygons):
                mask_polygon = self._simplify_polygon(polygons[idx])
            candidates.append(
                {
                    "id": track_id,
                    "track_id": track_id,
                    "bbox": list(bbox),
                    "score": round(max(0.0, min(1.0, float(score))), 3),
                    "class_id": class_id,
                    "class_name": self._class_name(class_id, names),
                    "source": "yolo",
                    "refined": False,
                    "mask_quality": 0.0,
                    "identity_score": 0.0,
                    "negative_margin": 0.0,
                    "motion_score": 0.0,
                    "is_distractor": False,
                    "mask_polygon": mask_polygon,
                }
            )
        return candidates

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model_ready": self.ready_model,
            "path": str(self.model_path) if self.model_path else "",
            "input_size": self.input_size,
            "conf": self.conf,
            "iou": self.iou,
            "last_error": self.last_error,
        }


def build_proposal_detector(config: dict[str, Any]) -> YoloProposalDetector | OpenCVProposalDetector:
    kind = str(config.get("models", {}).get("proposal", {}).get("type", "yolo11_trt")).lower()
    if kind.startswith("yolo"):
        return YoloProposalDetector(config)
    return OpenCVProposalDetector(config)
