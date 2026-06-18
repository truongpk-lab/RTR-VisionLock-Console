import numpy as np

from app.vision.backbones import BACKBONE_REGISTRY, ManagedTracker, build_backbone


def test_registry_exposes_all_tiers():
    assert {"opencv", "sam2_video", "uetrack", "uetrack_onnx"} <= set(BACKBONE_REGISTRY)


def test_unknown_backbone_name_falls_back_to_opencv():
    backbone = build_backbone("does-not-exist", {})
    assert backbone.source == "opencv"
    assert backbone.available is True


def test_deep_backbones_report_unavailable_without_torch():
    # On a machine without torch/weights, the deep adapters must not crash; they
    # report unavailable so the managed tracker can fall back.
    for name in ("uetrack",):
        backbone = build_backbone(name, {})
        assert backbone.available is False
        assert backbone.last_error


def test_managed_tracker_falls_back_to_opencv_when_engine_absent():
    for name in ("uetrack", "uetrack_onnx", "sam2_video"):
        tracker = ManagedTracker({}, backbone=name)
        assert tracker.source == "opencv"  # graceful fallback keeps the lock alive


def test_is_fallback_surfaces_silent_opencv_drop():
    # A deep backbone was requested but OpenCV is actually running -> fallback.
    deep = ManagedTracker({}, backbone="uetrack")
    assert deep.is_fallback is True
    assert deep.to_dict()["fallback"] is True
    # Explicitly requesting opencv is not a "fallback" surprise.
    plain = ManagedTracker({}, backbone="opencv")
    assert plain.is_fallback is False
    assert plain.to_dict()["fallback"] is False


def test_managed_tracker_default_backbone_is_opencv():
    tracker = ManagedTracker({"samurai": {"use_video_predictor": False}})
    assert tracker.source == "opencv"
    info = tracker.to_dict()
    assert info["requested"] == "auto"


def test_reinit_reseeds_without_crashing():
    # reinit re-seeds the existing tracker for every (re)lock. On a torch-less box
    # the deep request runs as the OpenCV fallback (cv2's init returns None -> a
    # falsey bool, which the lock flow tolerates); reinit must just not raise and
    # keep the fallback running.
    tracker = ManagedTracker({}, backbone="uetrack")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[50:90, 50:90] = (60, 200, 240)
    r1 = tracker.reinit(frame, (50, 50, 40, 40))
    r2 = tracker.reinit(frame, (60, 60, 40, 40))  # second re-seed
    assert isinstance(r1, bool) and isinstance(r2, bool)
    assert tracker.source == "opencv"


def test_warmup_is_noop_without_torch():
    tracker = ManagedTracker({}, backbone="uetrack")
    tracker.warmup((480, 640))  # must never raise
    assert tracker._warmed is True
    assert tracker.source == "opencv"
    tracker.warmup((480, 640))  # idempotent


def test_close_releases_backbone():
    tracker = ManagedTracker({}, backbone="uetrack")
    tracker.close()  # must never raise even without torch/CUDA
    assert tracker.backbone is None
