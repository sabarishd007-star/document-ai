@echo off
title DocuMind - AI Document Assistant
echo ========================================================
echo       DocuMind: AI Document Assistant Starting...
echo ========================================================
echo.
cd /d "%~dp0"
.\venv\Scripts\python.exe cli.py run --port 8000 --open
pause
