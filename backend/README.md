# RTR VisionLock Console Backend

FastAPI + OpenCV backend for realtime target selection, tracking, memory matching, and re-acquire.

## Jetson YOLO + SAM2 models

- Export the fine-tuned detector as TensorRT FP16 and place it at `backend/models/yolo11n_custom.engine`.
- Place the SAM2.1 Hiera Tiny checkpoint at `backend/models/sam2.1_hiera_tiny.pt`.
- Install optional Jetson runtime packages (`ultralytics`, `torch`, and `sam2`) on the device. Without them, the backend stays usable through OpenCV/GrabCut fallbacks.
- The default config uses YOLO-first selection, SAM2 box refinement, CUDA FP16, `detector_interval: 5`, and `sam_refine_interval: 8`.

## Run

```powershell
cd "D:\app\RTR VisionLock Console\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

## Test

```powershell
cd "D:\app\RTR VisionLock Console\backend"
.\.venv\Scripts\Activate.ps1
pytest
```
