# RTR VisionLock Console Backend

FastAPI + OpenCV backend for realtime target selection, tracking, memory matching, and re-acquire.

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
