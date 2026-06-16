from __future__ import annotations

import threading
import time
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None


class CameraSource:
    """Threaded latest-frame capture.

    A dedicated reader thread pulls frames from the device as fast as it streams
    and keeps only the newest one; the processing loop's ``read()`` then waits for
    the next fresh frame instead of synchronously draining several. This is the
    dominant FPS lever on a webcam: an uncompressed (YUYV) 720p stream saturates
    USB bandwidth and collapses to a few FPS, and draining N frames per loop
    multiplied that latency. Two fixes: stream MJPG (compressed, full rate) and
    capture on its own thread so device I/O never blocks the vision pipeline.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.capture = None
        self.active = False
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest = None  # newest BGR frame from the device
        self._seq = 0        # increments once per captured frame
        self._consumed = 0   # last seq handed out by read()
        self._failed = False
        self._stop = threading.Event()

    def open(self, source: int | str | None = None) -> bool:
        if cv2 is None:
            return False
        cfg = self.config.get("camera", {})
        source = cfg.get("source", 0) if source is None else source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            self.active = False
            return False
        # Request MJPG before geometry: an uncompressed 720p stream caps at a few
        # FPS over USB; MJPG streams compressed at the device's full rate.
        fourcc = str(cfg.get("fourcc", "MJPG")).upper()[:4]
        if fourcc:
            try:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:  # pragma: no cover - backend dependent
                pass
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("width", 1280)))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("height", 720)))
        capture.set(cv2.CAP_PROP_FPS, int(cfg.get("fps", 30)))
        # Smallest driver buffer so the reader thread always sees a fresh frame.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # pragma: no cover - backend dependent
            pass
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            self.active = False
            return False
        self.capture = capture
        with self._cond:
            self._latest = frame
            self._seq = 1
            self._consumed = 0
            self._failed = False
        self._stop.clear()
        self.active = True
        self._thread = threading.Thread(target=self._reader, name="visionlock-capture", daemon=True)
        self._thread.start()
        return True

    def _reader(self) -> None:
        while not self._stop.is_set():
            capture = self.capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                with self._cond:
                    self._failed = True
                    self._cond.notify_all()
                self.active = False
                break
            # cv2 allocates a new array per read, so publishing the reference is
            # race-free: the consumer's frame is never overwritten in place.
            with self._cond:
                self._latest = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        """Return the freshest captured frame, waiting briefly for a new one.

        Blocks only until the next frame arrives (or ``timeout``), never draining
        several, so processing is paced by the camera without piling up latency.
        """
        if cv2 is None:
            return False, None
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._consumed or self._failed, timeout=timeout)
            if self._seq != self._consumed and self._latest is not None:
                self._consumed = self._seq
                return True, self._latest
            if self._failed:
                return False, None
            # Camera alive but stalled past the timeout: hand back the last frame
            # so the UI stays warm instead of erroring on a transient hiccup.
            if self._latest is not None:
                return True, self._latest
            return False, None

    def close(self) -> None:
        self.active = False
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        time.sleep(0.02)
