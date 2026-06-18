"""Importing a local video and playing it back like a live camera.

Covers the two backend pieces of the feature: CameraSource opening a file path
(looping at the end so the feed never stops, paced to the file's FPS) and the
raw-body upload endpoint that stores the file for start_camera to open.
"""

import asyncio
import pathlib

import cv2
import numpy as np
from starlette.requests import Request

from app.main import upload_video
from app.vision.camera import CameraSource


def _write_clip(path: pathlib.Path, frames: int, fps: int = 30, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 17) % 255, np.uint8))
    writer.release()


def test_open_marks_file_and_paces_to_fps(tmp_path):
    clip = tmp_path / "clip.avi"
    _write_clip(clip, frames=5, fps=30)

    cam = CameraSource({})
    assert cam.open(str(clip)) is True
    try:
        assert cam._is_file is True
        # ~1/30s per frame, so a sane interval rather than racing through.
        assert 0.0 < cam._frame_interval < 0.1
    finally:
        cam.close()


def test_file_playback_loops_past_end(tmp_path):
    # Only 4 frames, but reading well beyond that must keep yielding frames
    # because the reader rewinds to the start instead of failing.
    clip = tmp_path / "short.avi"
    _write_clip(clip, frames=4, fps=60)

    cam = CameraSource({})
    assert cam.open(str(clip)) is True
    try:
        got = sum(1 for _ in range(12) if cam.read(timeout=2.0)[0])
    finally:
        cam.close()
    assert got >= 10


def test_open_missing_file_returns_false():
    cam = CameraSource({})
    assert cam.open("/no/such/video.mp4") is False
    assert cam.active is False


def _make_request(body: bytes, query: str) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "headers": [],
        "query_string": query.encode(),
    }
    return Request(scope, receive)


def test_upload_stores_bytes_and_returns_path():
    body = b"\x00\x01RTR-fake-video\x02\x03"
    result = asyncio.run(upload_video(_make_request(body, "filename=clip.mp4"), filename="clip.mp4"))
    saved = pathlib.Path(result["path"])
    try:
        assert result["name"] == "clip.mp4"
        assert saved.read_bytes() == body
    finally:
        saved.unlink(missing_ok=True)


def test_upload_strips_path_components_from_filename():
    result = asyncio.run(upload_video(_make_request(b"x", "filename=../../evil.mp4"), filename="../../evil.mp4"))
    saved = pathlib.Path(result["path"])
    try:
        assert result["name"] == "evil.mp4"
        assert saved.name == "evil.mp4"
    finally:
        saved.unlink(missing_ok=True)
