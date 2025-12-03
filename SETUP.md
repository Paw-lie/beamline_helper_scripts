# Beamline Helper Scripts - IQM Branch Setup Guide

## Quick Setup (Any Machine)

### Windows

**1. Create virtual environment:**
```powershell
python -m venv venv
```

**2. Activate:**
```powershell
.\venv\Scripts\Activate.ps1
```

**3. Install package in editable mode:**
```powershell
pip install -e .
```

**4. Run:**
```powershell
python beamline_helper_scripts\beamline_console_helper.py
```

---

### Linux/Mac

**1. Create virtual environment:**
```bash
python3 -m venv venv
```

**2. Activate:**
```bash
source venv/bin/activate
```

**3. Install package in editable mode:**
```bash
pip install -e .
```

**4. Run:**
```bash
python beamline_helper_scripts/beamline_console_helper.py
```

---

## Alternative: Install Dependencies Only

If you prefer not to install the package:

```bash
pip install -r requirements.txt
python beamline_helper_scripts/beamline_console_helper.py
```

---

## What's Included

This branch includes:
- **Q3_dev_workspace integration** for IQM slice analysis
- **Enhanced reconstruction tracking** with z-size reporting
- **New `analyze_slices` command** for CNR and sigma calculations
- **Multi-volume support** in reconstruction details

---

## Project Structure

```
beamline_helper_scripts_IQM_branch/
├── beamline_helper_scripts/     # Main scripts package
│   ├── __init__.py
│   └── beamline_console_helper.py
├── AiQuAM_270225/                # IQM analysis package
│   ├── __init__.py
│   ├── Q3_dev_workspace.py
│   ├── Q3_open_source_image_processing_functions.py
│   └── Q3_open_source_fit_two_half_gaussians.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Dependencies

- **numpy**: Array operations and volume data handling
- **Pillow**: Image creation for scan views
- **matplotlib**: Orthoslice visualization
- **scipy**: Scientific computing for Q3 analysis
- **scikit-image**: Image processing utilities
- **lmfit**: Profile fitting for IQM metrics

---

## Commands

Run the helper and use these commands:
- `check_detail` - Show all reconstructions with parameters
- `analyze_slices [identifier]` - Run IQM analysis on slice.vol files
- `scan_views [identifier]` - Create orthoslice visualizations
- `help` - Show all available commands

---

## Deactivate Virtual Environment

When done:
```bash
deactivate
```

---

## Troubleshooting

**Q3_dev_workspace import errors?**
- Make sure you're running from the project root directory
- Try: `pip install -e .` to install in editable mode

**Missing dependencies?**
- Re-run: `pip install -r requirements.txt`

**Permission errors on Windows?**
- Run PowerShell as Administrator or use: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
