# RTR VisionLock Console

RTR VisionLock Console là ứng dụng desktop dùng Electron + React cho giao diện và FastAPI + OpenCV cho backend tracking thời gian thực. Luồng mặc định là YOLO detect first: mở camera, bấm Select Target để hiện candidate boxes, click candidate, SAM2 refine mask, rồi identity-memory lock để bám đúng instance khi có vật thể tương tự.

## Yêu cầu

- Linux desktop, ví dụ Ubuntu/Debian.
- Visual Studio Code hoặc editor tương đương.
- Git.
- Python 3.12.x.
- Node.js LTS kèm `npm`, hoặc Node portable trong `.local-node/`.
- `wget` để bootstrap `pip` nếu venv chưa có sẵn.
- Webcam hoặc video source tương thích OpenCV.

Không cài thủ công thư viện frontend bằng `pip`. Backend dùng `backend/requirements.txt`; frontend/Electron dùng `frontend/package.json` và `frontend/package-lock.json`.

## Chạy nhanh trên Linux

Mở terminal tại thư mục project:

```bash
cd ~/code/RTR-VisionLock-Console
./start-app.sh
```

Lệnh này sẽ tạo venv backend, cài dependency Python, cài dependency frontend nếu thiếu, build UI và mở cửa sổ Electron. Backend tự chọn port trống, tự khởi động khi app mở và tự tắt khi đóng app.

Nếu file `.sh` chưa có quyền chạy:

```bash
chmod +x start-app.sh start-backend.sh start-frontend.sh
```

## Chạy riêng khi phát triển

Backend:

```bash
cd ~/code/RTR-VisionLock-Console
./start-backend.sh
```

Frontend dev server:

```bash
cd ~/code/RTR-VisionLock-Console
./start-frontend.sh
```

Frontend chạy tại `http://127.0.0.1:5173` và proxy `/api`, `/health`, `/ws` sang backend tại `http://127.0.0.1:8000`.

## Kiểm tra

Backend tests:

```bash
cd ~/code/RTR-VisionLock-Console/backend
.venv/bin/python -m pytest
```

Frontend build:

```bash
cd ~/code/RTR-VisionLock-Console/frontend
PATH="$PWD/../.local-node/bin:$PATH" npm run build
```

Kiểm tra backend health:

```bash
curl http://127.0.0.1:8000/health
```

## Cấu trúc

- `backend/`: FastAPI, OpenCV tracking, config, tests.
- `backend/app/core/`: state machine, metrics, logging, session runtime.
- `backend/app/vision/`: camera, proposal, segmenter, tracker, memory, Kalman, re-acquire.
- `backend/config/default.yaml`: camera, threshold, model slot và runtime config.
- `frontend/`: Vite React UI và Electron shell.
- `start-app.sh`: launcher desktop đầy đủ trên Linux.
- `start-backend.sh`: launcher backend trên Linux.
- `start-frontend.sh`: launcher frontend dev server trên Linux.
- `start-*.cmd`: launcher dành cho Windows.
- `setup.md`: hướng dẫn setup máy mới.

## Ghi chú runtime

- Camera mặc định là webcam index `0`. Đổi trong `backend/config/default.yaml` tại `camera.source` hoặc gọi `POST /api/camera/start` với `source`.
- Chọn mục tiêu mặc định là YOLO-first (`selection.mode: yolo`): YOLO phát hiện vật thể trước, người dùng click candidate, SAM2 refine vùng chọn, sau đó memory học positive/negative identity để tránh nhầm với vật thể tương tự.
- `POST /api/target/segment` chỉ là fallback/manual point mode; khi đang ở `CANDIDATE_TRACKING`, hãy click candidate hoặc gọi `POST /api/target/pick`.
- Nếu muốn Segment Anything/SAM2, đặt checkpoint vào `backend/models/` rồi bật `models.segmenter.enabled: true`. Khi thiếu model, backend fallback sang GrabCut/OpenCV và log rõ runtime fallback.
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

- `.\start-app.cmd: command not found`: đang chạy lệnh Windows trong bash. Dùng `./start-app.sh`.
- `cd: D:\app\...: No such file or directory`: đây là đường dẫn Windows. Trên Linux dùng thư mục project thật, ví dụ `~/code/RTR-VisionLock-Console`.
- `Permission denied` khi chạy `.sh`: chạy `chmod +x start-app.sh start-backend.sh start-frontend.sh`.
- `npm was not found`: cài Node.js LTS, hoặc dùng Node portable trong `.local-node/`.
- Không tạo được venv hoặc thiếu wheel OpenCV/ONNX Runtime: kiểm tra đang dùng Python 3.12.x, không dùng Python quá mới.
- Camera không mở: đóng app khác đang dùng webcam hoặc đổi `camera.source`.
