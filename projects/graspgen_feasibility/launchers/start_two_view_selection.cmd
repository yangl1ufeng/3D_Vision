@echo off
cd /d "%~dp0..\"
start "" "%~dp0..\.venv-vision\Scripts\pythonw.exe" "%~dp0..\select_grasp_target.py" --scene "%~dp0..\runs\new_desk\20261113_two_views_resolved\scene.json" %*
