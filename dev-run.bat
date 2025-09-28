@echo off
REM AutoDoc v2 Development Server Launcher
REM This script cleans cache and starts the development server

echo AutoDoc v2 Development Server
echo ==============================

REM Check if we're in the project root
if not exist "pyproject.toml" (
    echo Error: This script must be run from the project root directory
    echo Make sure you're in the autodoc-v2 folder
    pause
    exit /b 1
)

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo Warning: Virtual environment not detected
    echo Please activate your virtual environment first:
    echo   venv\Scripts\activate
    echo.
    pause
)

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "scripts\dev-run.ps1" %*
