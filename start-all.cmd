@echo off
REM RTR VisionLock Console now launches as a DESKTOP APP (not a web link).
REM This forwards to start-app.cmd, which builds the UI, starts the Python
REM backend automatically, and opens the Electron window.
call "%~dp0start-app.cmd"
