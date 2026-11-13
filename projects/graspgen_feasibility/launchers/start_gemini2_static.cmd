@echo off
setlocal
cd /d "%~dp0..\"
echo Keep the camera and desk still while 20 aligned frames are collected.
echo Close the live preview first; only one app can use the camera.
".venv-vision\Scripts\python.exe" -u capture_gemini_static.py
if errorlevel 1 (
  echo Static capture failed. Check the USB connection and close other camera apps.
  pause
  exit /b 1
)
".venv-vision\Scripts\python.exe" export_gemini_pointcloud.py
if errorlevel 1 (
  echo Depth was saved, but point cloud export failed.
  pause
  exit /b 1
)
echo Frozen depth saved under runs\gemini_static; point cloud saved under runs\gemini_pointcloud.
pause
