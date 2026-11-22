@echo off
cd /d "%~dp0"
title CAFA-TTA GUI
".venv\Scripts\python.exe" -u "gui\app.py"
if errorlevel 1 pause
