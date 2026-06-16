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
from app.vision.ekf import EKFGate
from app.vision.flow import CameraMotionEstimator
from app.vision.memory import MemoryBank
from app.vision.backbones import ManagedTracker
from app.vision.policy import ConfidenceManager, TrackMode
from app.vision.proposal import build_proposal_detector
from app.vision.reacquire import ReAcquireEngine
from app.vision.reacquire_controller import ReacquisitionController
from app.vision.segmenter import PromptableSegmenter
from app.vision.tracker import OpenCVTracker
from app.vision.utils import BBox, bbox_center, bbox_iou, clamp_bbox


class TrackingSession:
    def __init__(self) -> None:
        self.config = load_config()
        backend_root = Path(__file__).resolve().parents[2]
        self.logger = SessionLogger(backend_root / "logs", int(self.config.get("ui", {}).get("log_max_lines", 500)))
        self.state = TrackingState.INIT
        self.metrics = MetricState()
        self.camera = CameraSource(self.config)
        self.proposal = build_proposal_detector(self.config)
        self.segmenter = PromptableSegmenter(self.config)
        self.memory = MemoryBank(self.config)
        # EKF gate + camera ego-motion estimator: the optical-flow (tx,ty) is fed
        # into the gate each frame so camera shake/pan is not read as target drift.
        self.kalman = EKFGate(self.config)
        self.camera_motion = CameraMotionEstimator(self.config)
        self.reacquire = ReAcquireEngine(self.config, self.memory, self.kalman)
        # Stateful long-term tracking system. The tracker is just a module; the
        # deciding parts are confidence state + memory bank + global re-detection:
        #   Tier A NORMAL (LOCKED)    -> self.tracker        (high-confidence local track)
        #   Tier B REFIND (UNCERTAIN) -> self.refind_tracker (low-confidence local re-find)
        #   Tier C LOST               -> self.reacq          (global YOLO+ReID, confirm then re-lock)
        # Detection is OFF while tracking; it only runs when the manager says LOST.
        self.tracker = self._new_normal_tracker()
        self.refind_tracker = self._new_refind_tracker()
        self.confidence = ConfidenceManager(self.config)
        self.reacq = ReacquisitionController(self.config, self.memory, self.reacquire)
        self._last_good_bbox: BBox | None = None
        self.frame = None
        self.frame_jpeg = ""
        self.target_bbox: BBox | None = None
        self.kalman_bbox: BBox | None = None
        self.candidates: list[dict] = []
        self.selected_candidate_id: str | None = None
        self.target_class_id: int | None = None
        self.target_class_name: str = "object"
        # One lightweight tracker per candidate so candidate boxes follow the
        # scene in CANDIDATE_TRACKING instead of freezing on the proposal frame.
        self.candidate_trackers: dict[str, OpenCVTracker] = {}
        # Learning phase: an accurate tracker plus the appearance samples we
        # harvest each frame before committing the official lock.
        self.learning_tracker: OpenCVTracker | None = None
        self.learning_bbox: BBox | None = None
        self.learning_started_at: float = 0.0
        self.learning_samples: list = []
        self.learning_negative_samples: list = []
        self.prompt = ""
        self.timeline: list[dict] = []
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_frame_at = 0.0
        self._last_loop_at = 0.0
        self._frame_count = 0
        self.log("Core", "INFO", "RTR VisionLock backend initialized.")

    def log(self, module: str, level: str, message: str, **extra: Any) -> None:
        item = self.logger.add(module, level, message, self.state.value, **extra)
        if module in {"Core", "Tracker", "ReAcquire", "Proposal", "Memory", "Camera"}:
            self.timeline.append({"time": item["time"], "module": module, "level": level, "message": message})
            self.timeline = self.timeline[-80:]

    def _new_normal_tracker(self) -> ManagedTracker:
        """Tier A backbone (default UETrack-B; falls back to OpenCV when absent)."""
        name = self.config.get("tracking", {}).get("normal_backbone", "uetrack")
        return ManagedTracker(self.config, backbone=name)

    def _new_refind_tracker(self) -> ManagedTracker:
        """Tier B backbone (default EVPTrack; falls back to OpenCV when absent)."""
        name = self.config.get("tracking", {}).get("refind_backbone", "evptrack")
        return ManagedTracker(self.config, backbone=name)

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
            self.tracker = self._new_normal_tracker()
            self.refind_tracker = self._new_refind_tracker()
            self.confidence.reset()
            self.reacq.reset()
            self.camera_motion.reset()
            self._last_good_bbox = None
            self.log("Camera", "INFO", "Camera stopped.")
            return self.snapshot(include_frame=False)

    def select_target(self) -> dict:
        with self._lock:
            latest_frame = self.frame
            if latest_frame is None:
                self.log("Proposal", "WARN", "No frame available for target selection.")
                return self.snapshot(include_frame=False)
            # YOLO-first selection: generate detector boxes, then SAM2 refines
            # the chosen box before the learning window starts.
            candidates = [self._normalize_candidate(candidate, idx) for idx, candidate in enumerate(self.proposal.detect(latest_frame))]
            self.candidates = candidates
            self.selected_candidate_id = None
            self.target_class_id = None
            self.target_class_name = "object"
            self.metrics.candidates = len(candidates)
            # Spin up mini trackers for the strongest candidates (capped) so their
            # boxes follow the moving objects without paying for 20 trackers/frame.
            self._spawn_candidate_trackers(latest_frame, candidates)
            self.learning_tracker = None
            self.target_bbox = None
            self.kalman_bbox = None
            self.state = TrackingState.CANDIDATE_TRACKING
            proposal_status = self.proposal.to_dict() if hasattr(self.proposal, "to_dict") else {"backend": "unknown"}
            self.log("Proposal", "INFO", f"Candidates generated: {len(candidates)} via {proposal_status.get('backend')}; tracking live.")
            if not proposal_status.get("model_ready", False):
                self.log("Proposal", "INFO", f"YOLO runtime unavailable; using fallback proposer. {proposal_status.get('last_error', '')}".strip())
            return self.snapshot(include_frame=False)

    def segment_target(self, point: dict | None = None) -> dict:
        """Click-to-segment: SAM/GrabCut estimates a box at the click, then learns it.

        Works alongside YOLO candidate selection: a click that lands on a candidate
        is routed to ``pick_target`` by the UI, while a click on open scene comes
        here so the operator can grab anything the detector missed.
        """
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

    def select_box(self, bbox: list[int] | BBox | None = None) -> dict:
        """User dragged a box on the live frame -> learn that region directly.

        This is the classic single-object-tracking init (operator supplies the
        first box) and bypasses the detector entirely, so selection works even
        when YOLO is unavailable and candidates are noisy.
        """
        with self._lock:
            if self.frame is None or not bbox:
                self.log("Core", "WARN", "No frame/box available for box selection.")
                return self.snapshot(include_frame=False)
            x, y, w, h = (int(v) for v in bbox)
            if w < 8 or h < 8:
                self.log("Core", "WARN", "Selection box too small; drag a larger region.")
                return self.snapshot(include_frame=False)
            height, width = self.frame.shape[:2]
            clamped = clamp_bbox((x, y, w, h), width, height)
            self.log("Core", "INFO", "Target box selected; learning appearance.")
            return self._begin_learning(bbox=clamped)

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
        positive_point = self._positive_point_for_selection(bbox, point)
        negative_boxes = self._distractor_boxes(bbox)
        for negative in negative_boxes:
            self.memory.update_negative(self.frame, negative)
        refined = self.segmenter.refine_box(self.frame, bbox, positive_point=positive_point, negative_boxes=negative_boxes)
        if refined is not None:
            bbox = clamp_bbox(refined.bbox, width, height)
            self.metrics.mask_iou = refined.quality
            self._mark_selected_candidate_refined(bbox, refined.quality)
            if refined.refined:
                self.log("Proposal", "INFO", f"Selected box refined via {refined.backend}.", mask_quality=round(refined.quality, 3))
        # Accurate tracker dedicated to the chosen target for the learning window.
        self.learning_tracker = OpenCVTracker(self.config)
        ok = self.learning_tracker.init(self.frame, bbox)
        if not ok:
            self.log("Tracker", "WARN", "Learning tracker failed to init; using static bbox.")
        self.learning_bbox = bbox
        self.target_bbox = bbox
        self.learning_started_at = time.monotonic()
        self.learning_samples = []
        self.learning_negative_samples = [
            sample
            for sample in (self.memory.extract(self.frame, negative) for negative in negative_boxes)
            if sample is not None
        ]
        self.candidate_trackers = {}
        self.candidates = []
        self.metrics.candidates = 0
        self.state = TrackingState.LEARNING_TARGET
        self.log("Core", "INFO", "Target picked; learning appearance.")
        return self.snapshot(include_frame=False)

    def reset_tracking(self) -> dict:
        with self._lock:
            self.tracker = self._new_normal_tracker()
            self.refind_tracker = self._new_refind_tracker()
            self.confidence.reset()
            self.reacq.reset()
            self.camera_motion.reset()
            self._last_good_bbox = None
            self.target_bbox = None
            self.kalman_bbox = None
            self.candidates = []
            self._clear_selection_state()
            self.kalman.reset()
            self.metrics.track_score = 0.0
            self.metrics.confidence = 0.0
            self.metrics.similarity = 0.0
            self.metrics.mask_iou = 0.0
            self.metrics.motion = "IDLE"
            self.state = TrackingState.CAMERA_READY if self.camera.active else TrackingState.STOPPED
            self.log("Core", "INFO", "Tracking reset.")
            return self.snapshot(include_frame=False)

    def force_reacquire(self) -> dict:
        with self._lock:
            self.state = TrackingState.SEARCHING
            self.reacq.reset()
            self.log("ReAcquire", "INFO", "Initiating re-acquire sequence.")
        self._run_reacquire(force=True)
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
            self.proposal = build_proposal_detector(self.config)
            self.segmenter = PromptableSegmenter(self.config)
            self.kalman.max_error = float(self.config.get("thresholds", {}).get("kalman_max_error", 80))
            self.confidence.configure(self.config)
            self.reacq.configure(self.config)
            self.camera_motion = CameraMotionEstimator(self.config)
            self.log("Core", "INFO", "Runtime config patched.")
            return self.config

    def _normalize_candidate(self, candidate: dict, idx: int) -> dict:
        item = dict(candidate)
        candidate_id = str(item.get("id") or f"C{idx}")
        item.setdefault("id", candidate_id)
        item.setdefault("track_id", candidate_id)
        item.setdefault("score", 0.0)
        item.setdefault("class_id", None)
        item.setdefault("class_name", "object")
        item.setdefault("source", "opencv")
        item.setdefault("refined", False)
        item.setdefault("mask_quality", 0.0)
        item.setdefault("identity_score", 0.0)
        item.setdefault("negative_margin", 0.0)
        item.setdefault("motion_score", 0.0)
        item.setdefault("is_distractor", False)
        return item

    def _positive_point_for_selection(self, bbox: BBox, point: dict | None) -> tuple[int, int]:
        if point:
            return int(point.get("x", bbox[0] + bbox[2] // 2)), int(point.get("y", bbox[1] + bbox[3] // 2))
        return tuple(int(v) for v in bbox_center(bbox))

    def _same_candidate_class(self, left: dict, right: dict) -> bool:
        left_class = left.get("class_id")
        right_class = right.get("class_id")
        if left_class is not None and right_class is not None:
            return left_class == right_class
        return str(left.get("class_name", "object")) == str(right.get("class_name", "object"))

    def _candidate_distractor_boxes(self, selected: dict, candidates: list[dict]) -> list[BBox]:
        selected_bbox = tuple(selected["bbox"])
        sx, sy = bbox_center(selected_bbox)
        sw, sh = selected_bbox[2], selected_bbox[3]
        radius = float(self.config.get("sam2", {}).get("negative_prompt_radius", 2.5))
        max_distance = radius * max(1.0, (sw * sw + sh * sh) ** 0.5)
        distractors: list[BBox] = []
        for candidate in candidates:
            if candidate.get("id") == selected.get("id"):
                continue
            if not self._same_candidate_class(selected, candidate):
                continue
            bbox = tuple(candidate["bbox"])
            cx, cy = bbox_center(bbox)
            close = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5 <= max_distance
            touching = bbox_iou(selected_bbox, bbox) > 0.02
            if close or touching:
                candidate["is_distractor"] = True
                distractors.append(bbox)
        return distractors

    def _distractor_boxes(self, selected_bbox: BBox) -> list[BBox]:
        selected = None
        if self.selected_candidate_id:
            selected = next((item for item in self.candidates if item["id"] == self.selected_candidate_id), None)
        if selected is None:
            selected = {"id": "__selected__", "bbox": list(selected_bbox), "class_id": None, "class_name": "object"}
        return self._candidate_distractor_boxes(selected, self.candidates)

    def _enrich_candidates(self, frame, candidates: list[dict]) -> list[dict]:
        enriched = [self._normalize_candidate(candidate, idx) for idx, candidate in enumerate(candidates)]
        for candidate in enriched:
            bbox = tuple(candidate["bbox"])
            identity = self.memory.score(frame, bbox)
            motion = self.kalman.score_candidate(bbox)
            candidate["identity_score"] = identity["identity_score"]
            candidate["negative_margin"] = identity["negative_margin"]
            candidate["motion_score"] = round(motion, 3)
            candidate["reid_score"] = round(
                0.45 * candidate["identity_score"]
                + 0.25 * candidate["motion_score"]
                + 0.15 * float(candidate.get("score", 0.0))
                + 0.15 * float(candidate.get("mask_quality", 0.0)),
                3,
            )
        return enriched

    def _mark_selected_candidate_refined(self, bbox: BBox, quality: float) -> None:
        if not self.selected_candidate_id:
            return
        for candidate in self.candidates:
            if candidate["id"] == self.selected_candidate_id:
                candidate["bbox"] = list(bbox)
                candidate["refined"] = True
                candidate["mask_quality"] = round(quality, 3)
                return

    def _resolve_candidate(self, candidate_id: str | None, point: dict | None) -> BBox | None:
        if candidate_id:
            for candidate in self.candidates:
                if candidate["id"] == candidate_id:
                    self.selected_candidate_id = candidate_id
                    self.target_class_id = candidate.get("class_id")
                    self.target_class_name = str(candidate.get("class_name", "object"))
                    return tuple(candidate["bbox"])
        if point:
            px, py = int(point.get("x", -1)), int(point.get("y", -1))
            for candidate in self.candidates:
                x, y, w, h = candidate["bbox"]
                if x <= px <= x + w and y <= py <= y + h:
                    self.selected_candidate_id = candidate["id"]
                    self.target_class_id = candidate.get("class_id")
                    self.target_class_name = str(candidate.get("class_name", "object"))
                    return tuple(candidate["bbox"])
        if self.candidates:
            self.selected_candidate_id = self.candidates[0]["id"]
            self.target_class_id = self.candidates[0].get("class_id")
            self.target_class_name = str(self.candidates[0].get("class_name", "object"))
            return tuple(self.candidates[0]["bbox"])
        return None

    def _loop(self) -> None:
        self._last_loop_at = time.perf_counter()
        while not self._stop.is_set() and self.camera.active:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                with self._lock:
                    self.state = TrackingState.ERROR
                    self.metrics.camera = "OFFLINE"
                    self.log("Error", "ERROR", "Camera frame read failed.")
                break
            # Time only the processing work; the threaded camera already paces the
            # loop, so no manual FPS sleep is needed (it would just cap throughput).
            started = time.perf_counter()
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

    def _clear_selection_state(self) -> None:
        self.candidate_trackers = {}
        self.learning_tracker = None
        self.learning_bbox = None
        self.learning_samples = []
        self.learning_negative_samples = []
        self.learning_started_at = 0.0
        self.selected_candidate_id = None
        self.target_class_id = None
        self.target_class_name = "object"
        self.metrics.candidates = 0

    def _dispatch_frame(self, frame) -> None:
        if self.state == TrackingState.CANDIDATE_TRACKING:
            self._update_candidates(frame)
        elif self.state == TrackingState.LEARNING_TARGET:
            self._update_learning(frame)
        elif self.state in {TrackingState.SEARCHING, TrackingState.LOST}:
            # Stay in the global re-detect loop until a candidate is confirmed; the
            # tracker is idle here so detection is the only thing running.
            self._run_reacquire()
        else:
            self._update_tracking(frame)

    def _spawn_candidate_trackers(self, frame, candidates: list[dict]) -> None:
        """Create light mini trackers for the top-N candidates only (Jetson cap)."""
        cap = int(self.config.get("runtime", {}).get("max_candidate_trackers", 6))
        self.candidate_trackers = {}
        if cap <= 0 or not candidates:
            return
        ranked = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:cap]
        for candidate in ranked:
            mini = OpenCVTracker(self.config, light=True)
            if mini.init(frame, tuple(candidate["bbox"])):
                self.candidate_trackers[candidate["id"]] = mini

    def _rebuild_candidates(self, frame) -> None:
        """Re-run the detector during selection so candidate boxes stay fresh."""
        candidates = [self._normalize_candidate(c, idx) for idx, c in enumerate(self.proposal.detect(frame))]
        self.candidates = candidates
        self.metrics.candidates = len(candidates)
        self._spawn_candidate_trackers(frame, candidates)

    def _update_candidates(self, frame) -> None:
        """Advance candidate boxes: periodic re-detect plus mini-tracker follow."""
        interval = int(self.config.get("runtime", {}).get("candidate_redetect_interval", 15))
        if interval > 0 and self._frame_count > 0 and self._frame_count % interval == 0:
            self._rebuild_candidates(frame)
            return
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
                candidate["motion_score"] = 0.8
            else:
                candidate["motion_score"] = 0.0

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
        self.tracker = self._new_normal_tracker()
        ok = self.tracker.init(frame, bbox)
        # Seed the Tier B re-find tracker on the same lock so it already holds the
        # target template and can take over instantly when confidence drops.
        self.refind_tracker = self._new_refind_tracker()
        self.refind_tracker.init(frame, bbox)
        self.confidence.on_lock()
        self.reacq.reset()
        self.camera_motion.reset()
        self._last_good_bbox = bbox
        self.target_bbox = bbox
        self.kalman.reset(bbox)
        self.kalman_bbox = bbox
        if self.memory.load_samples(self.learning_samples, self.learning_negative_samples):
            self.log("Memory", "INFO", f"RAM/DRM seeded from {len(self.learning_samples)} learning samples.")
        elif not self.memory.initialize(frame, bbox):
            self.log("Memory", "WARN", "RAM initialization skipped; feature extraction failed.")
        else:
            self.log("Memory", "INFO", "RAM initialized from lock frame.")
        self.metrics.confidence = 0.85 if ok else 0.45
        self.metrics.similarity = 1.0
        self.metrics.track_score = track_score(self.metrics.confidence, 1.0, 1.0)
        self.metrics.motion = "STABLE"
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
        # Camera ego-motion compensation runs every frame (before update/predict)
        # so the EKF tracks the target independent of camera shake/pan.
        self.kalman.set_camera_motion(self.camera_motion.estimate(frame, self.target_bbox, self._frame_count))
        # Run ONLY the active tier's local tracker (normal in LOCKED, re-find in
        # UNCERTAIN). Detection stays OFF here; Tier C re-detect handles loss.
        tracker = self.tracker if self.confidence.mode == TrackMode.NORMAL else self.refind_tracker
        result = tracker.track(frame)
        ok, bbox = result.ok, result.bbox

        identity: dict | None = None
        motion_consistency = 0.0
        mask_quality = 0.0
        if not ok or bbox is None:
            score = 0.0
            self.metrics.confidence = 0.0
        else:
            height, width = frame.shape[:2]
            bbox = clamp_bbox(bbox, width, height)
            error, motion_consistency = self.kalman.update(bbox)
            identity = self.memory.score(frame, bbox)
            similarity = identity["identity_score"]
            confidence = self._confidence_from_jitter(bbox)
            # Blend in the backbone's own confidence when it reports one. OpenCV
            # reports a flat 1.0 (leaves this unchanged); deep trackers report a
            # real score that should pull confidence down as the target is lost.
            if result.affinity < 1.0:
                confidence = 0.5 * confidence + 0.5 * result.affinity
            mask_quality, refined_bbox = self._maybe_refine_target(frame, bbox, motion_consistency)
            score = track_score(confidence, similarity, motion_consistency, mask_quality)
            self.target_bbox = refined_bbox or bbox
            self._last_good_bbox = self.target_bbox
            self.kalman_bbox = self.kalman.predict()
            self.metrics.confidence = confidence
            self.metrics.similarity = similarity
            self.metrics.kalman_error = error
            self.metrics.motion = "STABLE" if motion_consistency > 0.7 else "DRIFT"
            self.metrics.mask_iou = mask_quality
            self.metrics.track_score = score

        gate = self.confidence.update(score, ok)
        self._apply_mode_transition(frame, gate.policy, bbox if ok else None)

        # Memory bank is tier-gated by the confidence manager: learn only on an
        # unambiguous LOCKED frame (SAMURAI admission still guards look-alikes),
        # then pace working/long-term consolidation. UNCERTAIN freezes the bank;
        # LOST never writes.
        if gate.allow_memory_update and ok and identity is not None:
            admission = self.memory.consider_update(
                frame, bbox, motion_consistency, affinity=mask_quality, scores=identity
            )
            if admission.admit:
                self.memory.consolidate(identity["negative_margin"])

        if gate.policy.state != self.state:
            self._log_state_transition(gate.policy.state, score)
        self.state = gate.policy.state
        if gate.run_detection:
            self._run_reacquire()

    def _apply_mode_transition(self, frame, decision, current_bbox: BBox | None) -> None:
        """Swap the active backbone when the policy crosses a tier boundary."""
        if decision.seed_refind:
            seed = current_bbox or self._last_good_bbox or self.kalman.predict()
            if seed is not None:
                self.refind_tracker = self._new_refind_tracker()
                if self.refind_tracker.init(frame, seed):
                    self.log("Tracker", "INFO", f"Confidence low -> re-find via {self.refind_tracker.source} (Tier B).")
        if decision.reinit_normal and self._last_good_bbox is not None:
            self.tracker = self._new_normal_tracker()
            self.tracker.init(frame, self._last_good_bbox)
            self.log("Tracker", "INFO", f"Recovered -> normal track via {self.tracker.source} (Tier A).")

    def _log_state_transition(self, state: TrackingState, score: float) -> None:
        if state == TrackingState.STABLE:
            self.log("Core", "INFO", "State -> STABLE.", track_score=score)
        elif state == TrackingState.UNCERTAIN:
            self.log("Core", "WARN", "State -> UNCERTAIN.", track_score=score)
        elif state == TrackingState.LOST:
            self.log("Core", "ERROR", "Target lost.")

    def _confidence_from_jitter(self, bbox: BBox) -> float:
        predicted = self.kalman.predict()
        if predicted is None:
            return 0.85
        px, py, pw, ph = predicted
        x, y, w, h = bbox
        jitter = abs(px - x) + abs(py - y) + 0.25 * (abs(pw - w) + abs(ph - h))
        norm = max(1.0, w + h)
        return clamp(1.0 - jitter / norm)

    def _refine_interval(self, motion_consistency: float) -> int:
        """Adaptive SAM2 cadence: stretch the interval while motion is calm.

        SAM2 is the heaviest op on Jetson. When the Kalman motion is consistent
        and the track is steady we rarely need a re-segmentation, so we refine far
        less often; when motion gets shaky we fall back to the dense interval.
        """
        runtime = self.config.get("runtime", {})
        base = max(1, int(runtime.get("sam_refine_interval", 8)))
        if not bool(runtime.get("adaptive_refine", True)):
            return base
        stable_motion = float(self.config.get("thresholds", {}).get("stable_motion", 0.7))
        if motion_consistency >= stable_motion:
            return max(base, int(runtime.get("stable_refine_interval", 30)))
        return base

    def _maybe_refine_target(self, frame, bbox: BBox, motion_consistency: float = 1.0) -> tuple[float, BBox | None]:
        # After lock, SAM2 mask refine during live tracking is OFF by default
        # (init-only) to protect the >=60 FPS budget and avoid a second algorithm
        # fighting the tracker. Flip runtime.refine_during_tracking to re-enable.
        if not self.config.get("runtime", {}).get("refine_during_tracking", False):
            return self.metrics.mask_iou or 1.0, None
        if not self.segmenter.ready_model:
            return self.metrics.mask_iou or 1.0, None
        interval = self._refine_interval(motion_consistency)
        if interval <= 0 or self._frame_count % interval != 0:
            return self.metrics.mask_iou or 1.0, None
        selected = {
            "id": "__target__",
            "bbox": list(bbox),
            "class_id": self.target_class_id,
            "class_name": self.target_class_name,
        }
        refined = self.segmenter.refine_box(
            frame,
            bbox,
            positive_point=tuple(int(v) for v in bbox_center(bbox)),
            negative_boxes=self._candidate_distractor_boxes(selected, self.candidates),
            motion_bbox=self.kalman.predict(),
        )
        if refined is None:
            return self.metrics.mask_iou or 1.0, None
        # Keep the tracker as the motion authority, but expose SAM2 consistency
        # and nudge the UI bbox when refinement is confident.
        if refined.quality >= float(self.config.get("thresholds", {}).get("mask_iou_threshold", 0.50)):
            return refined.quality, refined.bbox
        return refined.quality, None

    def _reacquire_due(self) -> bool:
        """Throttle global detection to detect_hz (2-5 Hz) while LOST."""
        detect_hz = float(self.config.get("reacquire", {}).get("detect_hz", 3))
        if detect_hz <= 0:
            return True
        fps = self.metrics.fps if self.metrics.fps > 1.0 else 30.0
        interval = max(1, int(round(fps / detect_hz)))
        return self._frame_count % interval == 0

    def _run_reacquire(self, force: bool = False) -> None:
        with self._lock:
            if self.frame is None:
                return
            self.state = TrackingState.SEARCHING
            # Detection runs only on a detect tick; off-tick frames just wait (the
            # confirmation buffer advances per detection tick, not per camera frame).
            if not force and not self._reacquire_due():
                return
            candidates = self._enrich_candidates(self.frame, self.proposal.detect(self.frame))
            candidates = self._refine_reacquire_candidates(self.frame, candidates)
            candidates = self._enrich_candidates(self.frame, candidates)
            self.candidates = candidates
            self.metrics.candidates = len(candidates)
            outcome = self.reacq.attempt(self.frame, candidates)
            if outcome.candidate is not None:
                self.metrics.similarity = float(outcome.candidate.get("similarity", self.metrics.similarity))
                self.metrics.confidence = outcome.reid_score
            if not outcome.confirmed or outcome.bbox is None:
                self.log(
                    "ReAcquire",
                    "INFO",
                    f"Re-detect confirming {outcome.confirming}/{outcome.need}.",
                    best=outcome.candidate,
                )
                return
            # Confirmed over enough consecutive detections -> commit the re-lock.
            bbox = tuple(outcome.bbox)
            self.tracker = self._new_normal_tracker()
            self.tracker.init(self.frame, bbox)
            self.refind_tracker = self._new_refind_tracker()
            self.refind_tracker.init(self.frame, bbox)
            self.confidence.on_lock()
            self.reacq.reset()
            self.camera_motion.reset()
            self._last_good_bbox = bbox
            self.target_bbox = bbox
            self.kalman.reset(bbox)
            self.kalman_bbox = bbox
            self.candidates = []  # detection off again once locked -> clear overlays
            self.metrics.candidates = 0
            self.metrics.track_score = outcome.reid_score
            self.log("ReAcquire", "INFO", "Candidate confirmed identity.", candidate=outcome.candidate)
            self.state = TrackingState.REACQUIRED
            self.log("Core", "SUCCESS", "Target re-acquired. State -> LOCKED_TRACKING.")
            self.state = TrackingState.LOCKED_TRACKING

    def _refine_reacquire_candidates(self, frame, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []
        max_refine = max(0, int(self.config.get("runtime", {}).get("max_sam_candidates", 3)))
        if max_refine == 0:
            return candidates
        ranked = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        refine_ids = {item["id"] for item in ranked[:max_refine]}
        refined_candidates = []
        for candidate in candidates:
            if candidate["id"] not in refine_ids:
                refined_candidates.append(candidate)
                continue
            result = self.segmenter.refine_box(
                frame,
                tuple(candidate["bbox"]),
                positive_point=tuple(int(v) for v in bbox_center(tuple(candidate["bbox"]))),
                negative_boxes=self._candidate_distractor_boxes(candidate, candidates),
                motion_bbox=self.kalman.predict(),
            )
            if result is None:
                refined_candidates.append(candidate)
                continue
            refined_candidates.append(
                {
                    **candidate,
                    "bbox": list(result.bbox),
                    "refined": result.refined,
                    "mask_quality": round(result.quality, 3),
                }
            )
        return refined_candidates

    def _encode_frame(self, frame) -> None:
        if cv2 is None:
            self.frame_jpeg = ""
            return
        # Process at full resolution but stream a smaller JPEG to the UI: cuts both
        # encode time and bandwidth. Overlays are positioned by percentage of
        # frame_size (kept at full res in the snapshot), so this is transparent.
        runtime = self.config.get("runtime", {})
        max_width = int(runtime.get("stream_max_width", 960))
        quality = int(runtime.get("stream_jpeg_quality", 70))
        stream = frame
        if max_width > 0 and frame.shape[1] > max_width:
            scale = max_width / float(frame.shape[1])
            stream = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", stream, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
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
                "proposal": self.proposal.to_dict() if hasattr(self.proposal, "to_dict") else {"backend": "unknown"},
                "segmenter": self.segmenter.to_dict(),
                "tracker": self.tracker.to_dict() if hasattr(self.tracker, "to_dict") else {"source": "unknown"},
                "tracking": {
                    "mode": self.confidence.mode.value,
                    "confidence_state": self.confidence.confidence_state,
                    "normal_backbone": self.tracker.source,
                    "refind_backbone": self.refind_tracker.source,
                    "reacquire": {
                        "confirming": self.reacq.buffer.streak,
                        "need": self.reacq.confirm_frames,
                        "detect_hz": float(self.config.get("reacquire", {}).get("detect_hz", 3)),
                    },
                },
                "metrics": self.metrics.to_dict(),
                "memory": self.memory.to_dict(),
                "logs": self.logger.latest(),
                "timeline": self.timeline[-40:],
                "prompt": self.prompt,
            }
