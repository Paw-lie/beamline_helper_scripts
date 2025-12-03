# Quick Start Guide

## For Fresh Installation

```bash
# 1. Clone the repository
git clone -b feature/iqm-branch https://github.com/Paw-lie/beamline_helper_scripts.git
cd beamline_helper_scripts

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\Activate.ps1  # Windows

# 3. Install the package
pip install -e .

# 4. Run the helper
python beamline_helper_scripts/beamline_console_helper.py
```

## For Existing python_venv

```bash
# Navigate to project
cd /path/to/beamline_helper_scripts

# Activate your existing venv
source python_venv/bin/activate  # Linux/Mac
# OR
.\python_venv\Scripts\Activate.ps1  # Windows

# Install the package (this fixes imports)
pip install -e .

# Run
python beamline_helper_scripts/beamline_console_helper.py
```

## What `pip install -e .` Does

- Installs the package in **editable mode**
- Sets up proper Python import paths
- Makes `AiQuAM_270225` and `beamline_helper_scripts` importable from anywhere
- No need to manually manage `sys.path`

## Verify Installation

```python
# Test imports
python -c "from AiQuAM_270225 import Q3_dev_workspace; print('✓ Q3 module OK')"
```

## Troubleshooting

**Still getting import errors?**

1. Make sure you're in the project root directory
2. Check that both folders exist:
   - `beamline_helper_scripts/`
   - `AiQuAM_270225/`
3. Reinstall: `pip uninstall beamline-helper-scripts -y && pip install -e .`

**No pandas/numpy/etc?**

```bash
pip install numpy scipy matplotlib Pillow scikit-image lmfit
```
