@echo off
cd /d "%~dp0..\"
"%~dp0..\.venv-sim\Scripts\python.exe" "%~dp0..\run_measured_grasp.py" %*
pause
