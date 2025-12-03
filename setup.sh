#!/bin/bash
# Setup script for Beamline Helper Scripts (IQM Branch) - Linux/Mac

echo "========================================"
echo "Beamline Helper Scripts - IQM Branch"
echo "Setup Script for Linux/Mac"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "[1/4] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing package in editable mode..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install package"
    exit 1
fi

echo "[4/4] Setup complete!"
echo ""
echo "========================================"
echo "To run the beamline helper:"
echo "  1. Activate: source venv/bin/activate"
echo "  2. Run: python beamline_helper_scripts/beamline_console_helper.py"
echo ""
echo "To deactivate: deactivate"
echo "========================================"
