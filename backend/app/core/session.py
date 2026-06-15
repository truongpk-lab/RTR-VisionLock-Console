from __future__ import annotations

import base64
import threading
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from app.core.config import deep_merge, load_config
from app.core.logger import SessionLogger
from app.core.metrics import MetricState, clamp, track_score
from app.core.states import TrackingState
from app.vision.camera import CameraSource
from app.vision.kalman import SimpleKalmanGate
from app.vision.memory import MemoryBank
from app.vision.proposal import OpenCVProposalDetector
from app.vision.reacquire import ReAcquireEngine
from app.vision.segmenter import PromptableSegmenter
from app.vision.tracker import OpenCVTracker
from app.vision.utils import BBox, clamp_bbox


class TrackingSession:
    def __init__(self) -> None:
        self.config = load_config()
        backend_root = Path(__file__).resolve().parents[2]
        self.logger = SessionLogger(backend_root / "logs", int(self.config.get("ui", {}).get("log_max_lines", 500)))
        self.state = TrackingState.INIT
        self.metrics = MetricState()
        self.camera = CameraSource(self.config)
        self.proposal = OpenCVProposalDetector(self.config)
        self.segmenter = PromptableSegmenter(self.config)
        self.memory = MemoryBank(self.config)
        self.kalman = SimpleKalmanGate(self.config)
        self.reacquire = ReAcquireEngine(self.config, self.memory, self.kalman)
        self.tracker = OpenCVTracker(self.config)
        self.frame = None
        self.frame_jpeg = ""
        self.target_bbox: BBox | None = None
        self.kalman_bbox: BBox | None = None
        self.candidates: list[dict] = []
        self.selected_candidate_id: str | None = None
        # One lightweight tracker per candidate so candidate boxes follow the
        # scene in CANDIDATE_TRACKING instead of freezing on the proposal frame.
        self.candidate_trackers: dict[str, OpenCVTracker] = {}
        # Learning phase: an accurate tracker plus the appearance samples we
        # harvest each frame before committing the official lock.
        self.learning_tracker: OpenCVTracker | None = None
        self.learning_bbox: BBox | None = None
        self.learning_started_at: float = 0.0
        self.learning_samples: list = []
        self.prompt = ""
        self.timeline: list[dict] = []
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lost_count = 0
        self._last_frame_at = 0.0
        self._last_loop_at = 0.0
        self._frame_count = 0
        self.log("Core", "INFO", "RTR VisionLock backend initialized.")

    def log(self, module: str, level: str, message: str, **extra: Any) -> None:
        item = self.logger.add(module, level, message, self.state.value, **extra)
        if module in {"Core", "Tracker", "ReAcquire", "Proposal", "Memory", "Camera"}:
            self.timeline.append({"time": item["time"], "module": module, "level": level, "message": message})
            self.timeline = self.timeline[-80:]

    def start_camera(self, source: int | str | None = None) -> dict:
        with self._lock:
            if self.camera.active:
                return self.snapshot(include_frame=False)
            opened = self.camera.open(source)
            if not opened:
                self.state = TrackingState.ERROR
                self.metrics.camera = "OFFLINE"
                self.log("Error", "ERROR", "Camera failed to start. Check source or OpenCV installation.")
                return self.snapshot(include_frame=False)
            self._stop.clear()
            self.state = TrackingState.CAMERA_READY
            self.metrics.camera = "ACTIVE"
            self.log("Camera", "INFO", "Camera started.")
            self._thread = threading.Thread(target=self._loop, name="visionlock-camera", daemon=True)
            self._thread.start()
            return self.snapshot(include_frame=False)

    def stop_camera(self) -> dict:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        with self._lock:
            self.camera.close()
            self.state = TrackingState.STOPPED
            self.metrics.camera = "OFFLINE"
            self.target_bbox = None
            self.kalman_bbox = None
            self.candidates = []
            self._clear_selection_state()
            self.tracker = OpenCVTracker(self.config)
            self.log("Camera", "INFO", "Camera stopped.")
            return self.snapshot(include_frame=False)

    def select_target(self) -> dict:
        with self._lock:
            latest_frame = self.frame
            if latest_frame is None:
                self.log("Proposal", "WARN", "No frame available for target selection.")
                return self.snapshot(include_frame=False)
            mode = str(self.config.get("selection", {}).get("mode", "point")).lower()
            if mode == "point":
                # Click-to-segment: wait for the user to click one object; SAM
                # (or GrabCut fallback) then returns a single coherent box.
                self.candidates = []
                self.candidate_trackers = {}
                self.selected_candidate_id = None
                self.metrics.candidates = 0
                self.learning_tracker = None
                self.target_bbox = None
                self.kalman_bbox = None
                self.state = TrackingState.POINT_PROMPT
                backend = self.segmenter.backend
                self.log("Proposal", "INFO", f"Click an object to segment (backend: {backend}).")
                if not self.segmenter.ready_model:
                    self.log("Proposal", "INFO", "SAM model not enabled; using GrabCut click-to-select fallback.")
                return self.snapshot(include_frame=False)
            # Legacy auto mode: multi-candidate proposal + per-candidate trackers.
            candidates = self.proposal.detect(latest_frame)
            self.candidates = candidates
            self.selected_candidate_id = None
            self.metrics.candidates = len(candidates)
            # Spin up one mini tracker per candidate, seeded on the proposal frame,
            # so their boxes track the moving objects from here on.
            self.candidate_trackers = {}
            for candidate in candidates:
                mini = OpenCVTracker(self.config, light=True)
                if mini.init(latest_frame, tuple(candidate["bbox"])):
                    self.candidate_trackers[candidate["id"]] = mini
            self.learning_tracker = None
            self.target_bbox = None
            self.kalman_bbox = None
            self.state = TrackingState.CANDIDATE_TRACKING
            self.log("Proposal", "INFO", f"Candidates generated: {len(candidates)}; tracking live.")
            if self.config.get("models", {}).get("proposal", {}).get("enabled") is not True:
                self.log("Proposal", "INFO", "OpenCV fallback active; FastSAM/MobileSAM model not enabled.")
            return self.snapshot(include_frame=False)

    def segment_target(self, point: dict | None = None) -> dict:
        """Click-to-segment: run SAM/GrabCut at the click -> LEARNING_TARGET."""
        with self._lock:
            if self.frame is None or point is None:
                self.log("Core", "WARN", "No frame/point available for segmentation.")
                return self.snapshot(include_frame=False)
            px, py = int(point.get("x", -1)), int(point.get("y", -1))
            bbox = self.segmenter.segment_point(self.frame, (px, py))
            if bbox is None:
                self.log("Proposal", "WARN", "Segmentation found no object at the click.")
                return self.snapshot(include_frame=False)
            self.log("Proposal", "INFO", f"Segmented object via {self.segmenter.backend}.")
            return self._begin_learning(bbox=bbox)

    def pick_target(self, candidate_id: str | None = None, point: dict | None = None) -> dict:
        """User clicked a (moving) candidate box -> enter LEARNING_TARGET."""
        with self._lock:
            return self._begin_learning(candidate_id, point)

    # Kept for API/back-compat: locking now always flows through the learning
    # phase, so this simply starts learning the chosen candidate.
    def lock_target(self, candidate_id: str | None = None, point: dict | None = None) -> dict:
        with self._lock:
            return self._begin_learning(candidate_id, point)

    def _begin_learning(
        self, candidate_id: str | None = None, point: dict | None = None, bbox: BBox | None = None
    ) -> dict:
        if self.frame is None:
            self.log("Core", "WARN", "Cannot pick target before a frame is available.")
            return self.snapshot(include_frame=False)
        if bbox is None:
            bbox = self._resolve_candidate(candidate_id, point)
        if bbox is None:
            self.log("Core", "WARN", "No candidate under cursor; run Select Target first.")
            return self.snapshot(include_frame=False)
        height, width = self.frame.shape[:2]
        bbox = clamp_bbox(bbox, width, height)
        # Accurate tracker dedicated to the chosen target for the learning window.
        self.learning_tracker = OpenCVTracker(self.config)
        ok = self.learning_tracker.init(self.frame, bbox)
        if not ok:
            self.log("Tracker", "WARN", "Learning tracker failed to init; using static bbox.")
        self.learning_bbox = bbox
        self.target_bbox = bbox
        self.learning_started_at = time.monotonic()
        self.learning_samples = []
        self.candidate_trackers = {}
        self.candidates = []
        self.metrics.candidates = 0
        self.state = TrackingState.LEARNING_TARGET
        self.log("Core", "INFO", "Target picked; learning appearance.")
        return self.snapshot(include_frame=False)

    def reset_tracking(self) -> dict:
        with self._lock:
            self.tracker = OpenCVTracker(self.config)
            self.target_bbox = None
            self.kalman_bbox = None
            self.candidates = []
            self._clear_selection_state()
            self.kalman.reset()
            self.metrics.track_score = 0.0
            self.metrics.confidence = 0.0
            self.metrics.similarity = 0.0
            self.metrics.motion = "IDLE"
            self.state = TrackingState.CAMERA_READY if self.camera.active else TrackingState.STOPPED
            self.log("Core", "INFO", "Tracking reset.")
            return self.snapshot(include_frame=False)

    def force_reacquire(self) -> dict:
        with self._lock:
            self.state = TrackingState.SEARCHING
            self.log("ReAcquire", "INFO", "Initiating re-acquire sequence.")
        self._run_reacquire()
        return self.snapshot(include_frame=False)

    def apply_prompt(self, prompt: str) -> dict:
        with self._lock:
            self.prompt = prompt.strip()
            self.log("UI", "INFO", f"Text prompt saved: {self.prompt or '(empty)'}.")
            if not self.config.get("models", {}).get("text_guided", {}).get("enabled", False):
                self.log("UI", "INFO", "Text prompt saved; text-guided detector not enabled.")
            return self.snapshot(include_frame=False)

    def patch_config(self, patch: dict[str, Any]) -> dict:
        with self._lock:
            self.config = deep_merge(self.config, patch)
            self.proposal = OpenCVProposalDetector(self.config)
            self.segmenter = PromptableSegmenter(self.config)
            self.kalman.max_error = float(self.config.get("thresholds", {}).get("kalman_max_error", 80))
            self.log("Core", "INFO", "Runtime config patched.")
            return self.config

    def _resolve_candidate(self, candidate_id: str | None, point: dict | None) -> BBox | None:
        if candidate_id:
            for candidate in self.candidates:
                if candidate["id"] == candidate_id:
                    self.selected_candidate_id = candidate_id
                    return tuple(candidate["bbox"])
        if point:
            px, py = int(point.get("x", -1)), int(point.get("y", -1))
            for candidate in self.candidates:
                x, y, w, h = candidate["bbox"]
                if x <= px <= x + w and y <= py <= y + h:
                    self.selected_candidate_id = candidate["id"]
                    return tuple(candidate["bbox"])
        if self.candidates:
            self.selected_candidate_id = self.candidates[0]["id"]
            return tuple(self.candidates[0]["bbox"])
        return None

    def _loop(self) -> None:
        self._last_loop_at = time.perf_counter()
        while not self._stop.is_set() and self.camera.active:
            started = time.perf_counter()
            ok, frame = self.camera.read()
            if not ok or frame is None:
                with self._lock:
                    self.state = TrackingState.ERROR
                    self.metrics.camera = "OFFLINE"
                    self.log("Error", "ERROR", "Camera frame read failed.")
                break
            with self._lock:
                self.frame = frame
                self._dispatch_frame(frame)
                self._encode_frame(frame)
                now = time.perf_counter()
                dt = max(1e-6, now - self._last_loop_at)
                self._last_loop_at = now
                self.metrics.fps = 0.85 * self.metrics.fps + 0.15 * (1.0 / dt) if self.metrics.fps else 1.0 / dt
                self.metrics.latency_ms = (time.perf_counter() - started) * 1000.0
                self._frame_count += 1
            target_fps = float(self.config.get("camera", {}).get("fps", 30))
            time.sleep(max(0.0, (1.0 / max(1.0, target_fps)) - (time.perf_counter() - started)))

    def _clear_selection_state(self) -> None:
        self.candidate_trackers = {}
        self.learning_tracker = None
        self.learning_bbox = None
        self.learning_samples = []
        self.learning_started_at = 0.0
        self.selected_candidate_id = None
        self.metrics.candidates = 0

    def _dispatch_frame(self, frame) -> None:
        if self.state == TrackingState.CANDIDATE_TRACKING:
            self._update_candidates(frame)
        elif self.state == TrackingState.LEARNING_TARGET:
            self._update_learning(frame)
        else:
            self._update_tracking(frame)

    def _update_candidates(self, frame) -> None:
        """Advance every candidate's mini tracker so its box follows the object."""
        if not self.candidate_trackers:
            return
        height, width = frame.shape[:2]
        for candidate in self.candidates:
            tracker = self.candidate_trackers.get(candidate["id"])
            if tracker is None:
                continue
            ok, bbox = tracker.update(frame)
            if ok and bbox is not None:
                candidate["bbox"] = list(clamp_bbox(bbox, width, height))

    def _crop_quality(self, frame, bbox: BBox) -> float:
        """Higher = sharper/contrastier crop. Used to gate learning samples."""
        if cv2 is None:
            return 0.0
        x, y, w, h = clamp_bbox(bbox, frame.shape[1], frame.shape[0])
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0 or w < 8 or h < 8:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _update_learning(self, frame) -> None:
        """LEARNING_TARGET: keep the target box live and harvest good samples."""
        learn_cfg = self.config.get("learning", {})
        duration = float(learn_cfg.get("duration_sec", 2.5))
        min_samples = int(learn_cfg.get("min_samples", 8))
        max_samples = int(learn_cfg.get("max_samples", 20))
        quality_threshold = float(learn_cfg.get("quality_threshold", 80.0))

        height, width = frame.shape[:2]
        bbox = self.learning_bbox
        if self.learning_tracker is not None:
            ok, tracked = self.learning_tracker.update(frame)
            if ok and tracked is not None:
                bbox = clamp_bbox(tracked, width, height)
        if bbox is None:
            return
        self.learning_bbox = bbox
        self.target_bbox = bbox  # UI draws the moving target box live.

        # Collect an appearance sample this frame if the crop looks clean.
        if len(self.learning_samples) < max_samples:
            quality = self._crop_quality(frame, bbox)
            if quality >= quality_threshold or not self.learning_samples:
                feature = self.memory.extract(frame, bbox)
                if feature is not None:
                    self.learning_samples.append(feature)

        elapsed = time.monotonic() - self.learning_started_at
        self.metrics.confidence = clamp(min(1.0, elapsed / max(0.1, duration)))
        self.metrics.similarity = clamp(len(self.learning_samples) / max(1, min_samples))
        self.metrics.motion = "LEARNING"

        enough_time = elapsed >= duration
        enough_samples = len(self.learning_samples) >= max_samples
        if (enough_time and len(self.learning_samples) >= 1) or enough_samples:
            self._finalize_lock(frame)

    def _finalize_lock(self, frame) -> None:
        """Commit the official lock: main tracker + RAM/DRM + Kalman from the
        stabilised learning bbox and harvested memory samples."""
        bbox = self.learning_bbox
        if bbox is None:
            self.log("Core", "WARN", "Lock aborted; no stable bbox from learning.")
            self.state = TrackingState.CAMERA_READY
            return
        height, width = frame.shape[:2]
        bbox = clamp_bbox(bbox, width, height)
        self.tracker = OpenCVTracker(self.config)
        ok = self.tracker.init(frame, bbox)
        self.target_bbox = bbox
        self.kalman.reset(bbox)
        self.kalman_bbox = bbox
        if self.memory.load_samples(self.learning_samples):
            self.log("Memory", "INFO", f"RAM/DRM seeded from {len(self.learning_samples)} learning samples.")
        elif not self.memory.initialize(frame, bbox):
            self.log("Memory", "WARN", "RAM initialization skipped; feature extraction failed.")
        else:
            self.log("Memory", "INFO", "RAM initialized from lock frame.")
        self.metrics.confidence = 0.85 if ok else 0.45
        self.metrics.similarity = 1.0
        self.metrics.track_score = track_score(self.metrics.confidence, 1.0, 1.0)
        self.metrics.motion = "STABLE"
        self._lost_count = 0
        self.learning_tracker = None
        self.learning_samples = []
        self.state = TrackingState.LOCKED_TRACKING
        self.log("Core", "SUCCESS", "Target locked. State -> LOCKED_TRACKING.")
        self.log("Tracker", "INFO", f"Main tracker initialized ({self.tracker.kind}).")

    def _update_tracking(self, frame) -> None:
        if self.state not in {
            TrackingState.LOCKED_TRACKING,
            TrackingState.TRACKING,
            TrackingState.STABLE,
            TrackingState.UNCERTAIN,
            TrackingState.REACQUIRED,
        }:
            return
        ok, bbox = self.tracker.update(frame)
        if not ok or bbox is None:
            self._lost_count += 1
            self.metrics.confidence = 0.0
        else:
            height, width = frame.shape[:2]
            bbox = clamp_bbox(bbox, width, height)
            error, motion_consistency = self.kalman.update(bbox)
            similarity = self.memory.similarity(frame, bbox)
            confidence = self._confidence_from_jitter(bbox)
            score = track_score(confidence, similarity, motion_consistency, 1.0)
            self.target_bbox = bbox
            self.kalman_bbox = self.kalman.predict()
            self.metrics.confidence = confidence
            self.metrics.similarity = similarity
            self.metrics.kalman_error = error
            self.metrics.motion = "STABLE" if motion_consistency > 0.7 else "DRIFT"
            self.metrics.mask_iou = 1.0
            self.metrics.track_score = score
            thresholds = self.config.get("thresholds", {})
            stable = float(thresholds.get("stable_threshold", 0.7))
            uncertain = float(thresholds.get("uncertain_threshold", 0.45))
            memory_update = float(thresholds.get("memory_update_threshold", 0.78))
            if score >= stable:
                if self.state != TrackingState.STABLE:
                    self.log("Core", "INFO", "State -> STABLE.", track_score=score)
                self.state = TrackingState.STABLE
                self._lost_count = 0
                if score >= memory_update:
                    self.memory.update_ram(frame, bbox)
            elif score >= uncertain:
                if self.state != TrackingState.UNCERTAIN:
                    self.log("Core", "WARN", "State -> UNCERTAIN.", track_score=score)
                self.state = TrackingState.UNCERTAIN
                self._lost_count += 1
            else:
                self._lost_count += 1
        lost_frames = int(self.config.get("thresholds", {}).get("lost_frames", 5))
        if self._lost_count >= lost_frames and self.state != TrackingState.SEARCHING:
            self.state = TrackingState.LOST
            self.log("Core", "ERROR", "Target lost.")
            self._run_reacquire()

    def _confidence_from_jitter(self, bbox: BBox) -> float:
        predicted = self.kalman.predict()
        if predicted is None:
            return 0.85
        px, py, pw, ph = predicted
        x, y, w, h = bbox
        jitter = abs(px - x) + abs(py - y) + 0.25 * (abs(pw - w) + abs(ph - h))
        norm = max(1.0, w + h)
        return clamp(1.0 - jitter / norm)

    def _run_reacquire(self) -> None:
        with self._lock:
            if self.frame is None:
                return
            self.state = TrackingState.SEARCHING
            candidates = self.proposal.detect(self.frame)
            self.candidates = candidates
            self.metrics.candidates = len(candidates)
            best = self.reacquire.best(self.frame, candidates)
            threshold = float(self.config.get("thresholds", {}).get("reacquire_threshold", 0.75))
            if best and best["reid_score"] >= threshold:
                bbox = tuple(best["bbox"])
                self.tracker = OpenCVTracker(self.config)
                self.tracker.init(self.frame, bbox)
                self.target_bbox = bbox
                self.kalman.reset(bbox)
                self.kalman_bbox = bbox
                self.metrics.similarity = best["similarity"]
                self.metrics.confidence = best["reid_score"]
                self.metrics.track_score = best["reid_score"]
                self._lost_count = 0
                self.log("ReAcquire", "INFO", "Candidate matched identity.", candidate=best)
                self.state = TrackingState.REACQUIRED
                self.log("Core", "SUCCESS", "Target re-acquired. State -> LOCKED_TRACKING.")
                self.state = TrackingState.LOCKED_TRACKING
            else:
                self.log("ReAcquire", "WARN", "No candidate passed re-acquire threshold.", best=best)

    def _encode_frame(self, frame) -> None:
        if cv2 is None:
            self.frame_jpeg = ""
            return
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        self.frame_jpeg = base64.b64encode(buffer).decode("ascii") if ok else ""

    def snapshot(self, include_frame: bool = True) -> dict:
        with self._lock:
            return {
                "app": self.config.get("app", {}).get("name", "RTR VisionLock Console"),
                "state": self.state.value,
                "frame": self.frame_jpeg if include_frame else "",
                "frame_size": list(self.frame.shape[1::-1]) if self.frame is not None else [0, 0],
                "target_bbox": list(self.target_bbox) if self.target_bbox else None,
                "kalman_bbox": list(self.kalman_bbox) if self.kalman_bbox else None,
                "candidate_boxes": self.candidates,
                "selected_candidate_id": self.selected_candidate_id,
                "learning": {
                    "active": self.state == TrackingState.LEARNING_TARGET,
                    "samples": len(self.learning_samples),
                    "elapsed": round(max(0.0, time.monotonic() - self.learning_started_at), 2)
                    if self.learning_started_at
                    else 0.0,
                    "duration": float(self.config.get("learning", {}).get("duration_sec", 2.5)),
                },
                "segmenter": self.segmenter.to_dict(),
                "metrics": self.metrics.to_dict(),
                "memory": self.memory.to_dict(),
                "logs": self.logger.latest(),
                "timeline": self.timeline[-40:],
                "prompt": self.prompt,
            }
