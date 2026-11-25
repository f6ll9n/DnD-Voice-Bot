@echo off
cd /d "%~dp0"
where ollama >nul 2>nul
if errorlevel 1 (
    echo [!] Ollama not found.
    echo Download it from https://ollama.com/download  ^(free^)
    echo Install, then run this file again.
    pause
    exit /b 1
)
set MODEL=%1
if "%MODEL%"=="" set MODEL=qwen2.5:3b
echo Pulling model: %MODEL%
ollama pull %MODEL%
echo.
echo Done. If you changed the model, put the same name into config.yaml -> llm.model
pause
