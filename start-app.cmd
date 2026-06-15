@echo off
setlocal

REM VSCode / Electron-based terminals export this and would make Electron run as
REM plain Node (no app window). Clear it so the desktop shell launches correctly.
set "ELECTRON_RUN_AS_NODE="

echo === RTR VisionLock Console (Desktop) ===

REM --- Backend virtual environment ---
cd /d "%~dp0backend"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Python 3.12 was not found through py launcher; falling back to py -3.
    py -3 -m venv .venv
  )
  if errorlevel 1 (
    echo py launcher failed; falling back to python from PATH.
    python -m venv .venv
  )
)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create backend virtual environment.
  pause
  exit /b 1
)
REM Only install backend deps when missing so re-launches open straight away.
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, cv2, onnxruntime, pydantic; legacy=getattr(cv2,'legacy',None); ok=hasattr(cv2,'TrackerCSRT_create') or hasattr(cv2,'TrackerKCF_create') or (legacy is not None and (hasattr(legacy,'TrackerCSRT_create') or hasattr(legacy,'TrackerKCF_create'))); raise SystemExit(0 if ok else 1)" >nul 2>nul
if errorlevel 1 (
  echo Installing backend dependencies...
  ".venv\Scripts\python.exe" -m pip uninstall -y opencv-python opencv-python-headless >nul 2>nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Backend dependency install failed.
    pause
    exit /b 1
  )
)

REM --- Frontend / desktop shell ---
cd /d "%~dp0frontend"
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo npm.cmd was not found. Install Node.js LTS, then open a new terminal.
  pause
  exit /b 1
)
if not exist "node_modules\electron" (
  echo Installing frontend and Electron dependencies...
  call npm.cmd install
  if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
  )
)

echo Building UI...
call npm.cmd run build
if errorlevel 1 (
  echo UI build failed.
  pause
  exit /b 1
)

echo Launching desktop application...
REM Call the Electron binary directly (not "npm run") so the app window always
REM opens; npm's wrapper can swallow the launch in some shells.
set "ELECTRON_RUN_AS_NODE="
if exist "node_modules\.bin\electron.cmd" (
  call "node_modules\.bin\electron.cmd" .
) else (
  call npm.cmd run electron
)

endlocal
