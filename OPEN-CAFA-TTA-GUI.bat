@echo off
setlocal
cd /d "%~dp0"
title CAFA-TTA GUI

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u "gui\app.py"
) else (
    echo No local virtual environment found. Using installed Python.
    python -u "gui\app.py"
)

if errorlevel 1 (
    echo.
    echo GUI failed. Install dependencies using:
    echo python -m pip install -r requirements.txt
    pause
)
