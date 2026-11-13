@echo off
cd /d "%~dp0..\"
if not exist ".venv-sim\Scripts\python.exe" (
  echo Simulation environment missing. See SIMULATION.md.
  pause
  exit /b 1
)
".venv-sim\Scripts\python.exe" run_simulation.py --gui --keep-open
if errorlevel 1 pause
