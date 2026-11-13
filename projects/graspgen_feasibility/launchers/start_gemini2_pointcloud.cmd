@echo off
setlocal
cd /d "%~dp0..\"
if not exist ".venv-vision\Scripts\python.exe" (
  echo Missing camera environment. See GEMINI2_CAMERA.md.
  pause
  exit /b 1
)
if "%~1"=="" (
  ".venv-vision\Scripts\python.exe" export_gemini_pointcloud.py
) else (
  ".venv-vision\Scripts\python.exe" export_gemini_pointcloud.py "%~1"
)
set "EXPORT_EXIT=%ERRORLEVEL%"
echo.
echo Point cloud saved under runs\gemini_pointcloud. Exit code: %EXPORT_EXIT%
pause
exit /b %EXPORT_EXIT%
