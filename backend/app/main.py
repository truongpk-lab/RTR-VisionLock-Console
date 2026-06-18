from __future__ import annotations

import asyncio
import os
import pathlib
import tempfile
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.core.session import TrackingSession

# Imported videos land here, then the camera opens them by path like any other
# source. Kept in the OS temp dir so it needs no setup and is cleaned by the OS.
UPLOAD_DIR = pathlib.Path(tempfile.gettempdir()) / "visionlock_uploads"


class CameraStartRequest(BaseModel):
    source: int | str | None = None


class LockTargetRequest(BaseModel):
    candidate_id: str | None = None
    point: dict[str, int] | None = None


class SegmentRequest(BaseModel):
    point: dict[str, int] | None = None


class BoxRequest(BaseModel):
    bbox: list[int] | None = None


class PromptRequest(BaseModel):
    prompt: str = ""


app = FastAPI(title="RTR VisionLock Console API")
session = TrackingSession()

# Local-only service consumed by the desktop shell (file:// / dynamic port) and
# the browser dev server, so allow any local origin without credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "RTR VisionLock Console"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    return session.snapshot(include_frame=False)


@app.post("/api/camera/upload")
async def upload_video(request: Request, filename: str = "video.mp4") -> dict[str, str]:
    """Store an uploaded video and return its path for /api/camera/start.

    The body is the raw file bytes (streamed to disk, so large clips never load
    fully into memory); this avoids a python-multipart dependency. The caller
    then runs it with start_camera(source=path), which opens it like a webcam.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = os.path.basename(filename) or "video.mp4"
    dest = UPLOAD_DIR / name
    with dest.open("wb") as out:
        async for chunk in request.stream():
            out.write(chunk)
    return {"path": str(dest), "name": name}


@app.post("/api/camera/start")
def start_camera(payload: CameraStartRequest | None = None) -> dict[str, Any]:
    return session.start_camera(payload.source if payload else None)


@app.post("/api/camera/stop")
def stop_camera() -> dict[str, Any]:
    return session.stop_camera()


@app.post("/api/target/select")
def select_target() -> dict[str, Any]:
    return session.select_target()


@app.post("/api/target/segment")
def segment_target(payload: SegmentRequest | None = None) -> dict[str, Any]:
    return session.segment_target(payload.point if payload else None)


@app.post("/api/target/box")
def select_box(payload: BoxRequest | None = None) -> dict[str, Any]:
    return session.select_box(payload.bbox if payload else None)


@app.post("/api/target/pick")
def pick_target(payload: LockTargetRequest | None = None) -> dict[str, Any]:
    return session.pick_target(payload.candidate_id if payload else None, payload.point if payload else None)


@app.post("/api/target/lock")
def lock_target(payload: LockTargetRequest | None = None) -> dict[str, Any]:
    return session.lock_target(payload.candidate_id if payload else None, payload.point if payload else None)


@app.post("/api/tracking/reset")
def reset_tracking() -> dict[str, Any]:
    return session.reset_tracking()


@app.post("/api/reacquire/force")
def force_reacquire() -> dict[str, Any]:
    return session.force_reacquire()


@app.post("/api/prompt/apply")
def apply_prompt(payload: PromptRequest) -> dict[str, Any]:
    return session.apply_prompt(payload.prompt)


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return session.config


@app.patch("/api/config")
def patch_config(payload: dict[str, Any]) -> dict[str, Any]:
    return session.patch_config(payload)


@app.post("/api/config/save")
def save_config() -> dict[str, Any]:
    return session.save_config()


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(session.snapshot(include_frame=True))
            # ~20 Hz: the frame JPEG is already encoded in the capture loop, so a
            # snapshot just re-reads it — cheap enough to keep selection responsive.
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.environ.get("RTR_HOST", "127.0.0.1")
    port = int(os.environ.get("RTR_PORT", "8000"))
    # Reload spawns a watcher subprocess that is hard to terminate cleanly when
    # the backend is managed by the desktop shell, so default it off there.
    reload = os.environ.get("RTR_RELOAD", "1") not in ("0", "false", "False", "")

    if reload:
        uvicorn.run("app.main:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port, log_level="warning")
