@echo off
setlocal
cd /d "%~dp0..\"
if not exist ".venv-vision\Scripts\python.exe" (
  echo Missing camera environment. See GEMINI2_CAMERA.md.
  pause
  exit /b 1
)
echo Opening Gemini 2 live preview. Press Q or Esc in the image window to close.
".venv-vision\Scripts\python.exe" view_gemini2.py
if errorlevel 1 (
  echo Camera preview could not start. Check that no other app is using the camera.
  pause
  exit /b 1
)
