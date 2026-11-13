@echo off
cd /d "%~dp0..\"
if "%~1"=="" (
  start "" "%~dp0..\.venv-vision\Scripts\pythonw.exe" "%~dp0..\select_grasp_target.py" --scene "%~dp0..\runs\new_desk\20261113_first_view\scene.json"
) else (
  start "" "%~dp0..\.venv-vision\Scripts\pythonw.exe" "%~dp0..\select_grasp_target.py" %*
)
