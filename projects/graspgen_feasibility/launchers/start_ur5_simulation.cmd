@echo off
cd /d "%~dp0..\"
if not exist ".venv-sim\Scripts\python.exe" (
  echo Simulation environment missing. See SIMULATION.md.
  pause
  exit /b 1
)
".venv-sim\Scripts\python.exe" -u run_simulation.py --robot ur5_robotiq140 --gui --keep-open
if errorlevel 1 pause
