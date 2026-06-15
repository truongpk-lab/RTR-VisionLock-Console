# Setup máy mới cho RTR VisionLock Console

Tài liệu này dành cho agent hoặc người dùng trên máy Windows mới chỉ có Visual Studio Code.

## 1. Cài công cụ nền

Cài các phần mềm sau, sau đó đóng toàn bộ terminal và mở lại:

- Git for Windows: https://git-scm.com/download/win
- Python 3.12.x: https://www.python.org/downloads/windows/
- Node.js LTS: https://nodejs.org/
- Visual Studio Code: https://code.visualstudio.com/

Khi cài Python, bật tùy chọn `Add python.exe to PATH` nếu có. Nếu dùng Python launcher, lệnh `py -3.12 --version` phải chạy được.

Extension VS Code nên có:

- Python
- Pylance
- ESLint
- GitLens hoặc GitHub Pull Requests

## 2. Kiểm tra môi trường

Mở PowerShell mới:

```powershell
git --version
py -3.12 --version
node --version
npm.cmd --version
```

Nếu `npm.ps1 cannot be loaded`, dùng `npm.cmd` thay cho `npm`.

## 3. Clone repo

```powershell
cd D:\app
git clone https://github.com/truongpk-lab/RTR-VisionLock-Console.git "RTR VisionLock Console"
cd "D:\app\RTR VisionLock Console"
```

Nếu đã tải ZIP thay vì clone Git, hãy giải nén vào `D:\app\RTR VisionLock Console`, rồi chạy:

```powershell
git init
git remote add origin https://github.com/truongpk-lab/RTR-VisionLock-Console.git
git fetch origin main
```

## 4. Chạy ứng dụng desktop

```powershell
cd "D:\app\RTR VisionLock Console"
.\start-app.cmd
```

Launcher sẽ tự:

- tạo `backend/.venv` bằng Python 3.12 nếu có;
- cài backend packages từ `backend/requirements.txt`;
- cài frontend/Electron packages từ `frontend/package-lock.json`;
- build UI;
- mở cửa sổ Electron và backend local.

## 5. Kiểm tra backend

```powershell
cd "D:\app\RTR VisionLock Console\backend"
.\.venv\Scripts\python.exe -m pytest
```

Kiểm tra OpenCV contrib tracker:

```powershell
cd "D:\app\RTR VisionLock Console\backend"
.\.venv\Scripts\python.exe -c "import cv2; legacy=getattr(cv2,'legacy',None); print(cv2.__version__); print(hasattr(cv2,'TrackerCSRT_create') or hasattr(cv2,'TrackerKCF_create') or (legacy is not None and (hasattr(legacy,'TrackerCSRT_create') or hasattr(legacy,'TrackerKCF_create'))))"
```

Dòng cuối phải in `True`.

## 6. Kiểm tra frontend

```powershell
cd "D:\app\RTR VisionLock Console\frontend"
npm.cmd ci
npm.cmd run build
```

## 7. Chạy dev mode

Backend:

```powershell
cd "D:\app\RTR VisionLock Console"
.\start-backend.cmd
```

Frontend browser dev:

```powershell
cd "D:\app\RTR VisionLock Console"
.\start-frontend.cmd
```

Desktop dev với hot reload:

```powershell
cd "D:\app\RTR VisionLock Console\frontend"
npm.cmd run dev
npm.cmd run app:dev
```

## 8. Commit và push

Đăng nhập GitHub bằng Git Credential Manager khi được hỏi, hoặc đăng nhập trong VS Code Source Control.

```powershell
cd "D:\app\RTR VisionLock Console"
git status
git add .
git status --short
git commit -m "chore: prepare reproducible Windows setup"
git fetch origin main
git merge origin/main --allow-unrelated-histories
git push -u origin main
```

Không commit các thư mục sinh tự động:

- `frontend/node_modules/`
- `frontend/dist/`
- `backend/.venv/`
- `backend/logs/`
- `__pycache__/`
- `.pytest_cache/`

## 9. Gỡ lỗi nhanh

- Thiếu `git`: cài Git for Windows rồi mở terminal mới.
- Thiếu `npm.cmd`: cài Node.js LTS rồi mở terminal mới.
- Backend install lỗi vì Python quá mới: cài Python 3.12.x và chạy lại.
- Webcam không lên: đóng app khác đang dùng camera hoặc đổi `camera.source` trong `backend/config/default.yaml`.
