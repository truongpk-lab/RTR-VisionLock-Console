@echo off
setlocal

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

echo Installing backend dependencies...
".venv\Scripts\python.exe" -m pip uninstall -y opencv-python opencv-python-headless >nul 2>nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Backend dependency install failed.
  pause
  exit /b 1
)

echo Starting backend at http://127.0.0.1:8000
".venv\Scripts\python.exe" -m app.main

pause
