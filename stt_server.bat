@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [!] Run install.bat first
    pause
    exit /b 1
)
echo This machine will serve speech recognition to the bot over the network.
echo Keep this window open while playing.
echo.
".venv\Scripts\python.exe" tools\stt_server.py %*
pause
