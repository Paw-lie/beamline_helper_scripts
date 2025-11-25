@echo off
REM Setup script for Beamline Helper Scripts (IQM Branch) - Windows

echo ========================================
echo Beamline Helper Scripts - IQM Branch
echo Setup Script for Windows
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/4] Setup complete!
echo.
echo ========================================
echo To run the beamline helper:
echo   1. Activate: venv\Scripts\activate.bat
echo   2. Run: python beamline_helper_scripts\beamline_console_helper.py
echo.
echo To deactivate: deactivate
echo ========================================
pause
