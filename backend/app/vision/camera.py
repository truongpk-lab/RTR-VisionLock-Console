from __future__ import annotations

import time
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None


class CameraSource:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.capture = None
        self.active = False

    def open(self, source: int | str | None = None) -> bool:
        if cv2 is None:
            return False
        cfg = self.config.get("camera", {})
        source = cfg.get("source", 0) if source is None else source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.active = False
            return False
        width = int(cfg.get("width", 1280))
        height = int(cfg.get("height", 720))
        fps = int(cfg.get("fps", 30))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.active = True
        return True

    def read(self):
        if not self.capture or not self.active:
            return False, None
        ok, frame = self.capture.read()
        if not ok:
            self.active = False
        return ok, frame

    def close(self) -> None:
        self.active = False
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        time.sleep(0.02)
