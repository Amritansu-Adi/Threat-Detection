@echo off
setlocal

REM Plays TEST videos with the same visual/audio/fusion/risk pipeline used by live mode.
REM Usage:
REM   run_test_video_demo.bat
REM   run_test_video_demo.bat TEST\any_video_name.mp4

cd /d C:\programing\MachineLearning\person_detect\yolo
set YOLO_CONFIG_DIR=%CD%\.ultralytics
set PYTHON_EXE=C:\Users\Amritansu Aditya\.conda\envs\threat-detect\python.exe

if not exist "%PYTHON_EXE%" (
  echo Could not find threat-detect Python at:
  echo %PYTHON_EXE%
  echo Edit run_test_video_demo.bat and set PYTHON_EXE to your environment.
  pause
  exit /b 1
)

if "%~1"=="" (
  "%PYTHON_EXE%" tools\demo_test_videos.py
) else (
  "%PYTHON_EXE%" tools\demo_test_videos.py --video "%~1"
)

pause
