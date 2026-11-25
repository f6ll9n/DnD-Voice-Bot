@echo off
cd /d "%~dp0"
echo Stopping DnD Voice Bot and Ollama (freeing RAM/VRAM)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -like 'python*' -and $_.CommandLine -like '*DnD-Voice-Bot*') -or $_.Name -like 'ollama*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
taskkill /F /IM ollama.exe >nul 2>nul
taskkill /F /IM "ollama app.exe" >nul 2>nul
echo Done. Bot and Ollama stopped, VRAM freed.
pause
