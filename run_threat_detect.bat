@echo off
REM Threat Detection System Launcher for Conda Environment
REM This script activates the conda threat-detect environment and runs the app

echo.
echo ====================================================
echo  Threat Detection System - Conda Environment
echo ====================================================
echo.

REM Call conda activation hook
call C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1

REM Activate threat-detect environment
call conda activate threat-detect

REM Change to project directory
cd /d C:\programing\MachineLearning\person_detect\yolo

REM Run the application
echo Running with Python: %PYTHON%
python main.py --verbose

REM Keep window open if there's an error
pause
