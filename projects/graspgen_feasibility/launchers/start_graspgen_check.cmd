@echo off
setlocal
cd /d "%~dp0..\"
if not exist ".venv-vision\Scripts\python.exe" (
  echo Missing .venv-vision. See GRASPGEN_RESTORED.md.
  pause
  exit /b 1
)
if "%~1"=="" (
  ".venv-vision\Scripts\python.exe" -u smoke_graspgen.py --scene "runs\virtual_camera\20261111_183228_541935\scene.json"
) else (
  ".venv-vision\Scripts\python.exe" -u smoke_graspgen.py --scene "%~1"
)
set "CHECK_EXIT=%ERRORLEVEL%"
echo.
echo Results are in runs\graspgen_restore. Exit code: %CHECK_EXIT%
pause
exit /b %CHECK_EXIT%
