# RTR VisionLock Console

RTR VisionLock Console là ứng dụng desktop Windows dùng Electron + React cho giao diện và FastAPI + OpenCV cho backend tracking thời gian thực. Ứng dụng hỗ trợ mở camera, chọn mục tiêu, khóa target, tracking, LOST/re-acquire, metrics, log và timeline qua WebSocket.

## Yêu cầu

- Windows 10/11.
- Visual Studio Code.
- Git for Windows.
- Python 3.12.x.
- Node.js LTS kèm `npm`.
- Webcam hoặc video source tương thích OpenCV.

Không cài thủ công thư viện frontend bằng `pip`. Backend dùng `backend/requirements.txt`; frontend/Electron dùng `frontend/package.json` và `frontend/package-lock.json`.

## Chạy nhanh

Mở PowerShell hoặc Command Prompt:

```powershell
cd "D:\app\RTR VisionLock Console"
.\start-app.cmd
```

Lệnh này sẽ tạo venv backend, cài dependency Python, cài dependency frontend nếu thiếu, build UI và mở cửa sổ Electron. Backend tự chọn port trống, tự khởi động khi app mở và tự tắt khi đóng app.

Có thể dùng lệnh tương đương:

```powershell
.\start-all.cmd
```

## Chạy riêng khi phát triển

Backend:

```powershell
cd "D:\app\RTR VisionLock Console"
.\start-backend.cmd
```

Frontend dev server:

```powershell
cd "D:\app\RTR VisionLock Console"
.\start-frontend.cmd
```

Frontend chạy tại `http://127.0.0.1:5173` và proxy `/api`, `/health`, `/ws` sang backend tại `http://127.0.0.1:8000`.

## Kiểm tra

Backend tests:

```powershell
cd "D:\app\RTR VisionLock Console\backend"
.\.venv\Scripts\python.exe -m pytest
```

Frontend build:

```powershell
cd "D:\app\RTR VisionLock Console\frontend"
npm.cmd run build
```

Kiểm tra backend health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Cấu trúc

- `backend/`: FastAPI, OpenCV tracking, config, tests.
- `backend/app/core/`: state machine, metrics, logging, session runtime.
- `backend/app/vision/`: camera, proposal, segmenter, tracker, memory, Kalman, re-acquire.
- `backend/config/default.yaml`: camera, threshold, model slot và runtime config.
- `frontend/`: Vite React UI và Electron shell.
- `start-app.cmd`: launcher desktop đầy đủ.
- `setup.md`: hướng dẫn setup máy mới.

## Ghi chú runtime

- Camera mặc định là webcam index `0`. Đổi trong `backend/config/default.yaml` tại `camera.source` hoặc gọi `POST /api/camera/start` với `source`.
- Chọn mục tiêu mặc định là click-to-segment (`selection.mode: point`). Khi chưa có SAM ONNX, backend dùng OpenCV GrabCut fallback nên app vẫn chạy được.
- Nếu muốn Segment Anything, đặt encoder/decoder ONNX vào `backend/models/` rồi bật `models.segmenter.enabled: true`.
- OpenCV dùng `opencv-contrib-python` để có tracker CSRT/KCF; không cài song song `opencv-python`.
- Log runtime ghi vào `backend/logs/session.jsonl` và không commit lên Git.

## API chính

- `GET /health`
- `GET /api/status`
- `POST /api/camera/start`
- `POST /api/camera/stop`
- `POST /api/target/select`
- `POST /api/target/segment`
- `POST /api/target/pick`
- `POST /api/target/lock`
- `POST /api/tracking/reset`
- `POST /api/reacquire/force`
- `POST /api/prompt/apply`
- `GET /api/config`
- `PATCH /api/config`
- `WS /ws/session`

## Lỗi thường gặp

- `npm.ps1 cannot be loaded`: dùng `npm.cmd` hoặc chạy file `.cmd`.
- `npm.cmd was not found`: cài Node.js LTS, đóng terminal, mở lại rồi chạy lại.
- Không tạo được venv hoặc thiếu wheel OpenCV/ONNX Runtime: kiểm tra đang dùng Python 3.12.x, không dùng Python quá mới.
- Camera không mở: đóng app khác đang dùng webcam hoặc đổi `camera.source`.
