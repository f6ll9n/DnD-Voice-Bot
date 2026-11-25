@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [!] .venv not found. Running install.bat...
    call install.bat
    if not exist ".venv\Scripts\python.exe" exit /b 1
)
".venv\Scripts\python.exe" tools\launcher.py --run
pause
