@echo off
setlocal

cd /d "%~dp0frontend"

where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo npm.cmd was not found. Install Node.js LTS, then open a new terminal.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo Installing frontend dependencies...
  npm.cmd install
  if errorlevel 1 (
    echo Frontend dependency install failed.
    pause
    exit /b 1
  )
)

echo Starting frontend at http://127.0.0.1:5173
npm.cmd run dev -- --host 127.0.0.1

pause
