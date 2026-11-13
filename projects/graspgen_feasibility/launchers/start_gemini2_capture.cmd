@echo off
setlocal
cd /d "%~dp0..\"
if not exist ".venv-vision\Scripts\python.exe" (
  echo Missing .venv-vision. See GEMINI2_CAMERA.md.
  pause
  exit /b 1
)
".venv-vision\Scripts\python.exe" -u probe_gemini2.py
set "CAPTURE_EXIT=%ERRORLEVEL%"
echo.
echo Capture saved under runs\gemini2_probe. Exit code: %CAPTURE_EXIT%
pause
exit /b %CAPTURE_EXIT%
