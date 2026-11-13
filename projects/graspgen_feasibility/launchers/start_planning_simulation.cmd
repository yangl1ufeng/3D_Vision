@echo off
cd /d "%~dp0..\"
".venv-sim\Scripts\python.exe" -u run_simulation.py --robot ur5_robotiq140 --planning --scenario barrier --gui --keep-open
if errorlevel 1 pause
