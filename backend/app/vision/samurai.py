"""SAMURAI-style main-target tracker (compatibility shim).

The backbone zoo and the managed wrapper now live in ``app.vision.backbones``;
this module keeps the historical ``SamuraiTracker`` / ``TrackResult`` names so
existing imports keep working. ``SamuraiTracker(config)`` is exactly
``ManagedTracker(config)`` with the legacy default-backbone selection (SAM2 video
predictor when ``samurai.use_video_predictor`` is on, else OpenCV).

The SAMURAI motion-aware mask selection + memory admission still live in
``app.vision.motion`` and run on top of whichever backbone is active.
"""

from __future__ import annotations

from app.vision.backbones import ManagedTracker, TrackResult

__all__ = ["SamuraiTracker", "TrackResult"]


class SamuraiTracker(ManagedTracker):
    """Back-compat alias: a managed tracker using the legacy default backbone."""
