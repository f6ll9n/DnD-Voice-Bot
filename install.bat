@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
    echo [!] Python not found in PATH.
    echo Install Python 3.12 from python.org
    echo and enable "Add python.exe to PATH" during setup.
    pause
    exit /b 1
)
python tools\setup_windows.py
pause
