import os
import sys
import time
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import Q3 analysis modules for slice analysis
try:
    # Add parent directory to path to import AiQuAM_270225
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from AiQuAM_270225 import Q3_dev_workspace
    Q3_AVAILABLE = True
except (ImportError, AttributeError) as e:
    Q3_AVAILABLE = False
    print(f"⚠️  Warning: Q3_dev_workspace module not available: {e}")
    print("    Slice analysis will be disabled.")

# ---------------------------------------------
# Beamline configuration
# ---------------------------------------------

BEAMLINE_CONFIGS = {
    "Anatomix": {
        "file_extensions": [".par", ".vol.info"],
        "param_keys": {
            "rotation_axis": "ROTATION_AXIS_POSITION",
            "paganin_value": "PAGANIN_Lmicron",
            "unsharp_coeff": "PUS",
            "unsharp_sigma": "PUC",
            "vol_size_x": "NUM_X",
            "vol_size_y": "NUM_Y",
            "vol_size_z": "NUM_Z",
            "camera_size_x_y": "NUM_IMAGE_1",
            "camera_size_z": "NUM_IMAGE_2",
            "voxel_size": "IMAGE_PIXEL_SIZE_1",
        }
    },
    "ESRF": {
        "file_extensions": [".vol.info", "infos.txt"],
        "param_keys": {
            "rotation_axis": "rotation_axis_position",
            "paganin_value": "delta_beta",
            "unsharp_coeff": "unsharp_coeff",
            "unsharp_sigma": "unsharp_sigma",
            "vol_size_x": "NUM_X",
            "vol_size_y": "NUM_Y",
            "vol_size_z": "NUM_Z",
            "voxel_size": "pixel_size_m",
        }
    }
}

# ---------------------------------------------
# Utility Functions
# ---------------------------------------------
def shorten_filename(filename, maxlen=24):
    """
    Shorten a filename for display, keeping the start and end, with [...] in the middle.
    """
    if len(filename) <= maxlen:
        return filename
    keep = maxlen // 2 - 2
    return f"{filename[:keep]}[...] {filename[-keep:]}"

def prompt_input(prompt, default=None):
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    try:
        value = input(prompt).strip()
    except EOFError:
        print("\nExiting.")
        sys.exit(0)
    return value if value else default

def get_beamline_profile(beamline):
    profile = BEAMLINE_CONFIGS.get(beamline)
    if not profile:
        print(f"/!\\  No configuration found for beamline '{beamline}'")
        return {}
    return profile

def parse_text_file(file_path):
    """
    Reads text file of key-value pairs.
    - Removes inline comments after #
    - Supports KEY=VALUE and KEY VALUE
    - Strips all spaces
    """
    data = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.split("#")[0].strip()  # remove comment
                if not line:
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                elif " " in line:
                    key, value = line.split(None, 1)
                else:
                    continue
                key = key.strip().replace(" ", "")
                value = value.strip().replace(" ", "")
                if key and value:
                    data[key] = value
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def extract_standard_params(settings, parsed_data):
    profile = get_beamline_profile(settings["beamline"])
    param_keys = profile.get("param_keys", {})
    mapped = {}
    for logical, actual in param_keys.items():
        value = parsed_data.get(actual)
        mapped[logical] = value
    return mapped

# ---------------------------------------------
# Commands
# ---------------------------------------------

def init_settings(existing_settings):
    print("\n--- Initialize Settings ---")
    new_settings = {}
    new_settings['beamline'] = prompt_input("Enter beamline", existing_settings.get('beamline'))
    new_settings['raw_data_path'] = prompt_input("Enter raw data path", existing_settings.get('raw_data_path'))
    new_settings['processed_data_path'] = prompt_input("Enter processed data path", existing_settings.get('processed_data_path'))

    # If ESRF, ask for camera size x-y and z
    if new_settings['beamline'] == "ESRF":
        try:
            cam_xy = prompt_input("Enter camera size X-Y (number of pixels, e.g. 2048)", existing_settings.get('camera_size_x_y'))
            cam_z = prompt_input("Enter camera size Z (number of pixels, e.g. 2048)", existing_settings.get('camera_size_z'))
            new_settings['camera_size_x_y'] = float(cam_xy) if cam_xy else None
            new_settings['camera_size_z'] = float(cam_z) if cam_z else None
        except Exception:
            print("/!\\  Invalid camera size. Setting to NaN.")
            new_settings['camera_size_x_y'] = float('nan')
            new_settings['camera_size_z'] = float('nan')

    print("\nSettings updated:")
    for key, val in new_settings.items():
        print(f"  {key}: {val}")
    return new_settings

def show_settings(settings):
    print("\n--- Current Settings ---")
    for key, val in settings.items():
        print(f"{key}: {val}")

def list_raw_files(settings):
    profile = get_beamline_profile(settings['beamline'])
    exts = profile.get('file_extensions', [])
    path = settings.get('raw_data_path', "")
    if not path or not os.path.isdir(path):
        print("/!\\  Invalid raw data path.")
        return
    print(f"\nListing files in {path} with extensions {exts}:")
    for fname in os.listdir(path):
        if any(fname.endswith(ext) for ext in exts):
            print(f" - {fname}")

def scan_parameters(settings):
    """
    Looks into processed_data_path/SCAN_ID/* for matching parameter files.
    Uses the first folder level as SCAN_ID.
    """
    profile = get_beamline_profile(settings["beamline"])
    exts = profile.get("file_extensions", [])
    processed_root = settings.get("processed_data_path", "")

    if not processed_root or not os.path.isdir(processed_root):
        print("/!\\  Invalid processed data path.")
        return

    # find first-level folders = scan IDs
    scan_ids = [d for d in os.listdir(processed_root)
                if os.path.isdir(os.path.join(processed_root, d))]
    if not scan_ids:
        print("/!\\  No scan folders found.")
        return

    for scan_id in scan_ids:
        scan_path = os.path.join(processed_root, scan_id)
        files = [f for f in os.listdir(scan_path) if any(f.endsWith(ext) for ext in exts)]
        if not files:
            print(f"\n/!\\  No matching files in scan {scan_id}")
            continue

        print(f"\nScan ID: {scan_id}")
        combined_data = {}
        for fname in files:
            full_path = os.path.join(scan_path, fname)
            parsed = parse_text_file(full_path)
            combined_data.update(parsed)

        mapped = extract_standard_params(settings, combined_data)
        print("Mapped Parameters:")
        for k, v in mapped.items():
            print(f"  {k}: {v}")

def check_reconstruction_status(settings):
    """
    Compare first-level folders in raw_data_path and processed_data_path.
    For each raw folder:
        - Check if processed folder exists
        - Check if any *.vol file exists with size > 0 bytes (anywhere in subfolders)
    Print both as ✔ or ✘ with aligned columns.
    Return dict with statuses.
    """
    raw_root = settings.get('raw_data_path', "")
    processed_root = settings.get('processed_data_path', "")

    if not os.path.isdir(raw_root) or not os.path.isdir(processed_root):
        print("/!\\  Both paths must be valid directories.")
        return

    raw_scans = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )
    processed_scans = set(
        d for d in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, d))
    )

    print("\n--- Reconstruction Status ---")
    print(f"{'SCAN_ID':<100} {'EXISTS':<7} {'.VOL':<7}")
    print("-" * 50)

    status = {}
    for scan in raw_scans:
        exists = scan in processed_scans
        has_vol = False
        if exists:
            scan_path = os.path.join(processed_root, scan)
            for root, dirs, files in os.walk(scan_path):
                for f in files:
                    if f.endswith('.vol'):
                        full_path = os.path.join(root, f)
                        try:
                            if os.path.getsize(full_path) > 0:
                                has_vol = True
                                break
                        except Exception:
                            continue
                if has_vol:
                    break
        exists_mark = "✔" if exists else "✘"
        vol_mark = "✔" if has_vol else "✘"
        print(f"{scan:<100} {exists_mark:<7} {vol_mark:<7}")
        status[scan] = {"exists": exists, "has_vol": has_vol}

    return status

def check_reconstruction_details(settings):
    """
    For each raw scan folder:
        - Check if processed folder exists
        - Gather all *.vol files (may be multiple) and check if any has size > 0
        - If exists, parse parameter files and map them (combined per scan)
        - Print key params (Paganin, unsharp_coeff, unsharp_sigma, rotation_axis)
        - List every volume file with its detected Z-size (from corresponding .info or .vol.info file)
    """
    def safe_str(v):
        return str(v) if v is not None else "NaN"

    raw_root = settings.get('raw_data_path', "")
    processed_root = settings.get('processed_data_path', "")
    beamline = settings.get('beamline')

    if not os.path.isdir(raw_root) or not os.path.isdir(processed_root):
        print("/!\\  Both paths must be valid directories.")
        return

    profile = get_beamline_profile(beamline)
    exts = profile.get("file_extensions", [])

    # Camera size and mode determination removed (no longer needed)

    raw_scans = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )
    processed_scans = set(
        d for d in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, d))
    )

    # Header
    print("\n--- Reconstruction Details ---")
    header = (
        f"{'SCAN_ID':<60} {'EXISTS':<7} {'.VOL':<7} "
        f"{'PAGANIN':<10} {'UNSH_COEFF':<12} {'UNSH_SIGMA':<12} "
        f"{'AXIS_POS':<10} {'Z_SIZE':<10} {'VOL_FILE':<40}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for scan in raw_scans:
        exists = scan in processed_scans
        has_vol = False
        params = None
        paganin = "NaN"
        unsharp_coeff = "NaN"
        unsharp_sigma = "NaN"
        axis_position = "NaN"
        vol_file_infos = []  # list of {file, z_size}

        if exists:
            scan_path = os.path.join(processed_root, scan)

            # Gather all .vol files
            vol_files = []
            for root, dirs, files in os.walk(scan_path):
                for f in files:
                    if f.endswith('.vol'):
                        vol_files.append(os.path.join(root, f))
            # Determine if any volume file has size > 0
            for vf in vol_files:
                try:
                    if os.path.getsize(vf) > 0:
                        has_vol = True
                        break
                except Exception:
                    continue

            # Parse all parameter files (combined per scan)
            combined_data = {}
            for root, dirs, files in os.walk(scan_path):
                for f in files:
                    if any(f.endswith(ext) for ext in exts):
                        file_path = os.path.join(root, f)
                        parsed = parse_text_file(file_path)
                        combined_data.update(parsed)

            if combined_data:
                params = extract_standard_params(settings, combined_data)
                paganin_val = params.get('paganin_value')
                try:
                    paganin = f"{float(paganin_val):.2f}"
                except (TypeError, ValueError):
                    paganin = safe_str(paganin_val)
                unsharp_coeff = safe_str(params.get('unsharp_coeff'))
                unsharp_sigma = safe_str(params.get('unsharp_sigma'))
                axis_position = safe_str(params.get('rotation_axis'))

            # Extract per-volume z-size info
            for vf in sorted(vol_files):
                z_size = "NaN"
                # Possible info file variants
                info_candidates = [vf + ".info", vf + ".vol.info"]
                for ic in info_candidates:
                    if os.path.isfile(ic):
                        parsed_info = parse_text_file(ic)
                        raw_z = parsed_info.get("NUM_Z") or parsed_info.get("vol_size_z")
                        if raw_z is not None:
                            try:
                                z_size = str(int(float(raw_z)))
                            except Exception:
                                z_size = safe_str(raw_z)
                        break
                vol_file_infos.append({"file": vf, "z_size": z_size})

        exists_mark = "Y" if exists else "N"
        vol_mark = "Y" if has_vol else "N"

        # Print result line
        axis_display = axis_position
        try:
            axis_display = f"{float(axis_position):.2f}"
        except Exception:
            axis_display = axis_position

        # If there are volume files, print one line per file; otherwise single line
        if vol_file_infos:
            first = True
            for info in vol_file_infos:
                file_display = shorten_filename(os.path.basename(info["file"]), 38)
                print(
                    f"{(scan if first else ''):<60} {exists_mark if first else '':<7} {vol_mark if first else '':<7} "
                    f"{(paganin if first else ''):<10} {(unsharp_coeff if first else ''):<12} {(unsharp_sigma if first else ''):<12} "
                    f"{(axis_display if first else ''):<10} {info['z_size']:<10} {file_display:<40}"
                )
                first = False
        else:
            print(
                f"{scan:<60} {exists_mark:<7} {vol_mark:<7} "
                f"{paganin:<10} {unsharp_coeff:<12} {unsharp_sigma:<12} {axis_display:<10} {'-':<10} {'-':<40}"
            )

        results[scan] = {
            "exists": exists,
            "has_vol": has_vol,
            "parameters": params,
            "paganin": paganin,
            "unsharp_coeff": unsharp_coeff,
            "unsharp_sigma": unsharp_sigma,
            "rotation_axis": axis_position,
            "volumes": vol_file_infos
        }

    return results

def show_pending_reconstructions(settings):
    """
    Show all scan folders that still need to be reconstructed:
    - If folder in raw but not in processed: status 'MISSING'
    - If folder in processed but no .vol file: check last file update, if <5min: 'ONGOING', else 'STALLED'
    Only print those that are not fully reconstructed.
    """
    raw_root = settings.get('raw_data_path', "")
    processed_root = settings.get('processed_data_path', "")

    if not os.path.isdir(raw_root) or not os.path.isdir(processed_root):
        print("/!\\  Both paths must be valid directories.")
        return

    now = time.time()
    five_minutes = 5 * 60

    raw_scans = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )
    processed_scans = set(
        d for d in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, d))
    )

    print("\n--- Pending Reconstructions ---")
    print(f"{'SCAN_ID':<100} {'STATUS':<3} {'LAST_UPDATE':<20}")
    print("-" * 70)

    for scan in raw_scans:
        if scan not in processed_scans:
            # Black large circle for missing (⬤)
            print(f"{scan:<100} {'⬤':<3} {'-':<20}")
        else:
            scan_path = os.path.join(processed_root, scan)
            has_vol = False
            latest_mtime = 0
            for root, dirs, files in os.walk(scan_path):
                for f in files:
                    if f.endswith('.vol'):
                        has_vol = True
                        break
                    try:
                        mtime = os.path.getmtime(os.path.join(root, f))
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                    except Exception:
                        continue
            if not has_vol:
                if latest_mtime == 0:
                    # Question mark for unknown
                    status_emoji = "?"
                    last_update = "-"
                else:
                    age = now - latest_mtime
                    if age < five_minutes:
                        # Yellow circle for ongoing
                        status_emoji = "o"
                    else:
                        # Red circle for stalled
                        status_emoji = "r"
                    last_update = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{scan:<100} {status_emoji:<3} {last_update:<20}")

def show_pending_details(settings):
    """
    Show detailed status for all scan folders:
    -  (black) for missing (not started)
    - o for ongoing (<5min) or incomplete .vol files (with reason)
    - r for stalled (>5min)
    - ? for unknown (no files found)
    - Y for done (.vol exists, >0GB, z-size matches camera z)
    Multiple .vol files: print multiple dots.
    Also shows slice.vol files as their own row.
    """
    raw_root = settings.get('raw_data_path', "")
    processed_root = settings.get('processed_data_path', "")
    beamline = settings.get('beamline')

    if not os.path.isdir(raw_root) or not os.path.isdir(processed_root):
        print("/!\\  Both paths must be valid directories.")
        return

    now = time.time()
    five_minutes = 5 * 60

    # Camera z-size
    camera_z = settings.get('camera_size_z')
    try:
        camera_z = int(camera_z)
    except Exception:
        camera_z = None

    raw_scans = sorted(
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    )
    processed_scans = set(
        d for d in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, d))
    )

    # Legend
    print("\n--- Pending Reconstructions (Detailed) ---")
    print("Legend:")
    print("   (black) for missing (not started)")
    print("  o for ongoing or incomplete .vol file. Size or file-size does not match expectations.")
    print("  r for stalled (>5min)")
    print("  ? for unknown (no files found)")
    print("  Y for done (.vol exists, >0GB, z-size matches camera z)")
    print("Each .vol and slice.vol file is shown as a separate row.\n")
    print(f"{'SCAN_ID':<60} {'FILE':<20} {'STATUS':<40} {'LAST_UPDATE':<20}")
    print("-" * 140)

    for scan in raw_scans:
        scan_path = os.path.join(processed_root, scan)
        if scan not in processed_scans:
            print(f"{scan:<60} {'-':<20} {'⬤':<40} {'-':<20}")
            continue

        # Gather all .vol and slice.vol files
        vol_files = []
        slice_vol_files = []
        latest_mtime = 0
        for root, dirs, files in os.walk(scan_path):
            for f in files:
                if f.endswith('.vol') and not f.endswith('slice.vol'):
                    vol_files.append(os.path.join(root, f))
                elif f == 'slice.vol':
                    slice_vol_files.append(os.path.join(root, f))
                # Track latest mtime for stalled/ongoing
                try:
                    mtime = os.path.getmtime(os.path.join(root, f))
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                except Exception:
                    continue

        # Show .vol files
        for vf in sorted(vol_files):
            status_emojis = ""
            yellow_reasons = []
            last_update = "-"
            try:
                size_gb = os.path.getsize(vf) / (1024 ** 3)
            except Exception:
                size_gb = 0
            # Try to get z-size from .vol.info or similar
            z_size = None
            info_file = vf + ".info"
            if os.path.isfile(info_file):
                params = parse_text_file(info_file)
                z_size = params.get("NUM_Z") or params.get("vol_size_z")
                try:
                    z_size = int(z_size)
                except Exception:
                    z_size = None
            # Check if file is >0GB and z-size matches
            if size_gb <= 0:
                status_emojis += "o"
                yellow_reasons.append("file empty")
            elif camera_z is not None and z_size != camera_z:
                status_emojis += "o"
                yellow_reasons.append(f"z-size {z_size} != camera {camera_z}")
            else:
                status_emojis += "Y"
            try:
                mtime = os.path.getmtime(vf)
                last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                last_update = "-"
            reason_str = ""
            if "o" in status_emojis and yellow_reasons:
                unique_reasons = list(dict.fromkeys(yellow_reasons))
                reason_str = " (" + ", ".join(unique_reasons) + ")"
            print(f"{scan:<60} {shorten_filename(os.path.basename(vf)):<24} {status_emojis:<40} {last_update:<20}{reason_str}")

        # Show slice.vol files
        for svf in sorted(slice_vol_files):
            status_emojis = ""
            yellow_reasons = []
            last_update = "-"
            try:
                size_gb = os.path.getsize(svf) / (1024 ** 3)
            except Exception:
                size_gb = 0
            # Try to get z-size from .vol.info or similar
            z_size = None
            info_file = svf + ".info"
            if os.path.isfile(info_file):
                params = parse_text_file(info_file)
                z_size = params.get("NUM_Z") or params.get("vol_size_z")
                try:
                    z_size = int(z_size)
                except Exception:
                    z_size = None
            # Check if file is >0GB and z-size matches
            if size_gb <= 0:
                status_emojis += "o"
                yellow_reasons.append("file empty")
            elif camera_z is not None and z_size != camera_z:
                status_emojis += "o"
                yellow_reasons.append(f"z-size {z_size} != camera {camera_z}")
            else:
                status_emojis += "Y"
            try:
                mtime = os.path.getmtime(svf)
                last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                last_update = "-"
            reason_str = ""
            if "🟡" in status_emojis and yellow_reasons:
                unique_reasons = list(dict.fromkeys(yellow_reasons))
                reason_str = " (" + ", ".join(unique_reasons) + ")"
            print(f"{scan:<60} {shorten_filename(os.path.basename(svf)):<24} {status_emojis:<40} {last_update:<20}{reason_str}")

        # If no .vol or slice.vol files found, show stalled/ongoing/unknown
        if not vol_files and not slice_vol_files:
            if latest_mtime == 0:
                status_emojis = "?"
                last_update = "-"
            else:
                age = now - latest_mtime
                if age < five_minutes:
                    status_emojis = "o"
                    yellow_reasons = ["recently updated, reconstruction ongoing"]
                else:
                    status_emojis = "r"
                last_update = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")
            reason_str = ""
            if "o" in status_emojis and yellow_reasons:
                unique_reasons = list(dict.fromkeys(yellow_reasons))
                reason_str = " (" + ", ".join(unique_reasons) + ")"
            print(f"{scan:<60} {'-':<20} {status_emojis:<40} {last_update:<20}{reason_str}")

def show_scan_details(settings, identifier):
    """
    Print all extracted parameters for every scan folder containing the identifier.
    Searches recursively for descriptive files in each matching folder.
    Sorts matching folders alphanumerically.
    """
    profile = get_beamline_profile(settings.get("beamline", ""))
    exts = profile.get("file_extensions", [])
    processed_root = settings.get("processed_data_path", "")

    if not processed_root or not os.path.isdir(processed_root):
        print("/!\\  Invalid processed data path.")
        return

    # Find all scan folders containing the identifier, sorted alphanumerically
    try:
        scan_ids = sorted(
            [d for d in os.listdir(processed_root)
             if os.path.isdir(os.path.join(processed_root, d)) and identifier in d],
            key=lambda x: x.lower()
        )
    except Exception as e:
        print(f"/!\\  Error listing processed data path: {e}")
        return

    if not scan_ids:
        print(f"/!\\  No scan folders found containing '{identifier}'.")
        return

    for scan_id in scan_ids:
        scan_path = os.path.join(processed_root, scan_id)
        found_files = []
        # Recursively search for files with the correct extensions
        for root, dirs, files in os.walk(scan_path):
            for f in files:
                if any(f.endswith(ext) for ext in exts):
                    found_files.append(os.path.join(root, f))
        if not found_files:
            print(f"\n/!\\  No matching files in scan {scan_id}")
            continue

        print(f"\nScan ID: {scan_id}")
        combined_data = {}
        for full_path in found_files:
            parsed = parse_text_file(full_path)
            combined_data.update(parsed)

        mapped = extract_standard_params(settings, combined_data)
        print("Mapped Parameters:")
        for k, v in mapped.items():
            print(f"  {k}: {v}")

def init_views_folder(settings):
    """
    Ask user for a folder to save scan view images to, and store in settings.
    """
    folder = prompt_input("Enter folder to save scan view images", settings.get('views_folder', ''))
    if not folder:
        print("No folder provided.")
        return settings
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        except Exception as e:
            print(f"/!\\  Could not create folder: {e}")
            return settings
    settings['views_folder'] = folder
    print(f"Views folder set to: {folder}")
    return settings

def create_scan_views(settings):
    """
    For each scan in processed_data_path, create a PNG view from the .vol file using memory mapping.
    Only create if PNG does not exist or is older than the .vol file.
    """
    processed_root = settings.get("processed_data_path", "")
    views_folder = settings.get("views_folder", "")

    if not processed_root or not os.path.isdir(processed_root):
        print("/!\\  Invalid processed data path.")
        return
    if not views_folder or not os.path.isdir(views_folder):
        print("/!\\  Invalid views folder. Please run 'init_views' first.")
        return

    # Get beamline profile for extension mapping
    profile = get_beamline_profile(settings.get("beamline", ""))
    exts = profile.get("file_extensions", [])
    param_keys = profile.get("param_keys", {})

    # Find all scan folders
    scan_ids = [d for d in os.listdir(processed_root)
                if os.path.isdir(os.path.join(processed_root, d))]
    if not scan_ids:
        print("/!\\  No scan folders found.")
        return

    for scan_id in scan_ids:
        scan_path = os.path.join(processed_root, scan_id)
        vol_files = []
        for root, dirs, files in os.walk(scan_path):
            for f in files:
                if f.endswith('vol.vol'):
                    vol_files.append(os.path.join(root, f))
        if not vol_files:
            continue

        for vol_file in vol_files:
            # Compose output PNG path
            png_name = f"{scan_id}_{os.path.basename(vol_file).replace('.vol', '')}.png"
            png_path = os.path.join(views_folder, png_name)

            # Check if PNG exists and is up-to-date
            if os.path.isfile(png_path):
                png_mtime = os.path.getmtime(png_path)
                vol_mtime = os.path.getmtime(vol_file)
                if png_mtime > vol_mtime:
                    print(f"View already exists and is up-to-date for {vol_file}")
                    continue

            # Find any descriptive file in the same folder as the .vol file
            info_file = None
            info_dir = os.path.dirname(vol_file)
            for candidate in os.listdir(info_dir):
                print(candidate)
                if any(candidate.endswith(ext) for ext in exts):
                    info_file = os.path.join(info_dir, candidate)
                    break
            if not info_file:
                print(f"/!\\  No info file found for {vol_file}, skipping.")
                continue

            params = parse_text_file(info_file)
            # Use the same logic as extract_standard_params to get the correct keys
            try:
                shape_z = int(params.get(param_keys.get("vol_size_z", "NUM_Z")) or params.get("NUM_Z") or params.get("vol_size_z"))
                shape_y = int(params.get(param_keys.get("vol_size_y", "NUM_Y")) or params.get("NUM_Y") or params.get("vol_size_y"))
                shape_x = int(params.get(param_keys.get("vol_size_x", "NUM_X")) or params.get("NUM_X") or params.get("vol_size_x"))
            except Exception:
                print(f"/!\\  Could not determine shape for {vol_file}, skipping.")
                continue

            dtype = np.float32  # or np.uint16, adjust as needed
            try:
                # Memory-map the volume file
                vol = np.memmap(vol_file, dtype=dtype, mode='r', shape=(shape_z, shape_y, shape_x))
                # Take a central slice along z
                slice_img = vol[shape_z // 2, :, :]
                # Normalize to 0-255 for PNG
                slice_norm = (slice_img - np.min(slice_img))
                if np.max(slice_norm) > 0:
                    slice_norm = slice_norm / np.max(slice_norm)
                slice_uint8 = (slice_norm * 255).astype(np.uint8)
                img = Image.fromarray(slice_uint8)
                img.save(png_path)
                print(f"Saved view: {png_path}")
                del vol
            except Exception as e:
                print(f"/!\\  Error creating view for {vol_file}: {e}")

def create_ortho_slices(settings, identifier=None):
    """
    For each .vol file in processed_data_path, create an orthoslice PNG (XY, XZ, YZ) in the views_folder.
    If identifier is given, only process scan folders containing the identifier.
    """
    processed_root = settings.get("processed_data_path", "")
    views_folder = settings.get("views_folder", "")

    if not processed_root or not os.path.isdir(processed_root):
        print("/!\\  Invalid processed data path.")
        return

    if not views_folder or not os.path.isdir(views_folder):
        print("/!\\  Invalid or missing views folder. Please select a folder.")
        settings = init_views_folder(settings)
        views_folder = settings.get("views_folder", "")
        if not views_folder or not os.path.isdir(views_folder):
            print("/!\\  Views folder still invalid. Aborting.")
            return

    profile = get_beamline_profile(settings.get("beamline", ""))
    param_keys = profile.get("param_keys", {})

    scan_ids = sorted([
        d for d in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, d)) and (identifier is None or identifier in d)
    ])
    if not scan_ids:
        print("/!\\  No scan folders found." + (f" (filtered by '{identifier}')" if identifier else ""))
        return

    try:
        for scan_id in scan_ids:
            scan_path = os.path.join(processed_root, scan_id)
            vol_files = []
            for root, dirs, files in os.walk(scan_path):
                for f in sorted(files):
                    if f.endswith('.vol') and not f.endswith('slice.vol'):
                        vol_files.append(os.path.join(root, f))
            for vol_file in vol_files:
                base = os.path.splitext(os.path.basename(vol_file))[0]
                out_file = os.path.join(views_folder, f"{scan_id}_{base}_ortho.png")
                if os.path.isfile(out_file):
                    png_mtime = os.path.getmtime(out_file)
                    vol_mtime = os.path.getmtime(vol_file)
                    if png_mtime > vol_mtime:
                        print(f"Skipping {out_file} (already up-to-date)")
                        continue

                info_file = vol_file + ".info"
                if not os.path.isfile(info_file):
                    print(f"/!\\  No info file for {vol_file}, skipping.")
                    continue

                params = parse_text_file(info_file)
                try:
                    shape_z = int(params.get(param_keys.get("vol_size_z", "NUM_Z")) or params.get("NUM_Z") or params.get("vol_size_z"))
                    shape_y = int(params.get(param_keys.get("vol_size_y", "NUM_Y")) or params.get("NUM_Y") or params.get("vol_size_y"))
                    shape_x = int(params.get(param_keys.get("vol_size_x", "NUM_X")) or params.get("NUM_X") or params.get("vol_size_x"))
                except Exception:
                    print(f"/!\\  Could not determine shape for {vol_file}, skipping.")
                    continue

                dtype = np.float32
                try:
                    raw = np.memmap(vol_file, dtype=dtype, mode='r')
                    data = np.reshape(raw, (shape_x, shape_y, shape_z), order='F')  # Fortran order

                    cx, cy, cz = shape_x // 2, shape_y // 2, shape_z // 2
                    image_downsample_factor = 6
                    slice_xy = data[:, :, cz][::image_downsample_factor, ::image_downsample_factor]
                    slice_xz = data[:, cy, :][::image_downsample_factor, ::image_downsample_factor]
                    slice_yz = data[cx, :, :][::image_downsample_factor, ::image_downsample_factor]

                    def norm(img):
                        img = img.astype(np.float32)
                        img -= np.min(img)
                        if np.max(img) > 0:
                            img /= np.max(img)
                        return img

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(norm(slice_xy).T, cmap='gray', origin='lower')
                    axes[0].set_title(f'XY (Z={cz})')
                    axes[1].imshow(norm(slice_xz).T, cmap='gray', origin='lower')
                    axes[1].set_title(f'XZ (Y={cy})')
                    axes[2].imshow(norm(slice_yz).T, cmap='gray', origin='lower')
                    axes[2].set_title(f'YZ (X={cx})')
                    for ax in axes:
                        ax.axis('off')
                    plt.tight_layout()

                    plt.savefig(out_file, dpi=150)
                    plt.close()
                    print(f"Saved ortho slices: {out_file}")
                    del raw
                except Exception as e:
                    print(f"Error processing {vol_file}: {e}")
    except KeyboardInterrupt:
        print("\n⏹  Ortho slice creation interrupted by user. Returning to command prompt.")
        return

def scan_views(settings):
    """
    For every .vol file, print its location, last update, and corresponding info file.
    If a views folder is set, create a PNG view for each volume if missing or outdated.
    """
    processed_root = settings.get("processed_data_path", "")
    views_folder = settings.get("views_folder", "")

    if not processed_root or not os.path.isdir(processed_root):
        print("/!\\  Invalid processed data path.")
        return

    # Prompt for views_folder if missing or invalid
    if not views_folder or not os.path.isdir(views_folder):
        print("/!\\  Invalid or missing views folder. Please select a folder.")
        settings = init_views_folder(settings)
        views_folder = settings.get("views_folder", "")
        if not views_folder or not os.path.isdir(views_folder):
            print("/!\\  Views folder still invalid. Aborting.")
            return

    # Get beamline profile for extension mapping
    profile = get_beamline_profile(settings.get("beamline", ""))
    exts = profile.get("file_extensions", [])
    param_keys = profile.get("param_keys", {})

    # Find all scan folders
    scan_ids = [d for d in os.listdir(processed_root)
                if os.path.isdir(os.path.join(processed_root, d))]
    if not scan_ids:
        print("/!\\  No scan folders found.")
        return

    found_any = False

    for scan_id in scan_ids:
        #print(scan_id)
        scan_path = os.path.join(processed_root, scan_id)
        for root, dirs, files in os.walk(scan_path):
            for f in files:
                if f.endswith('.vol') and not f.endswith('slice.vol'):
                    #print(f"Found volume file: {f} in {root}")
                    found_any = True
                    vol_file = os.path.join(root, f)
                    vol_mtime = os.path.getmtime(vol_file)
                    # --- Always use .vol.info as info file ---
                    info_file = vol_file + ".info"
                    if not os.path.isfile(info_file):
                        continue

                    # --- Scan view creation ---
                    if views_folder and os.path.isdir(views_folder):
                        base = os.path.splitext(f)[0]
                        png_name = f"{scan_id}_{base}.png"
                        png_path = os.path.join(views_folder, png_name)
                        create_view = False
                        overwrite = False
                        if not os.path.isfile(png_path):
                            create_view = True
                        else:
                            png_mtime = os.path.getmtime(png_path)
                            if png_mtime <= vol_mtime:
                                create_view = True
                                overwrite = True
                        if create_view:
                            params = parse_text_file(info_file)
                            try:
                                shape_z = int(params.get(param_keys.get("vol_size_z", "NUM_Z")) or params.get("NUM_Z") or params.get("vol_size_z"))
                                shape_y = int(params.get(param_keys.get("vol_size_y", "NUM_Y")) or params.get("NUM_Y") or params.get("vol_size_y"))
                                shape_x = int(params.get(param_keys.get("vol_size_x", "NUM_X")) or params.get("NUM_X") or params.get("vol_size_x"))
                            except Exception:
                                continue
                            dtype = np.float32  # or np.uint16, adjust as needed
                            try:
                                vol = np.memmap(vol_file, dtype=dtype, mode='r', shape=(shape_z, shape_y, shape_x))
                                slice_img = vol[shape_z // 2, :, :]
                                slice_norm = (slice_img - np.min(slice_img))
                                if np.max(slice_norm) > 0:
                                    slice_norm = slice_norm / np.max(slice_norm)
                                slice_uint8 = (slice_norm * 255).astype(np.uint8)
                                img = Image.fromarray(slice_uint8)
                                img.save(png_path)
                                del vol
                                if overwrite:
                                    print(f"Overwrote view: {png_path} (newer .vol found)")
                                else:
                                    print(f"Created view: {png_path}")
                            except Exception:
                                continue

    if not found_any:
        print("No .vol files found. Please use the 'check' or 'pending' command to see the state of the volume reconstruction.")

def analyze_slices(settings, identifier=None):
    """
    Analyze all slice.vol files using Q3_dev_workspace.slice_quick_IQM.
    For each slice.vol file:
        - Load the slice data
        - Apply slice_quick_IQM analysis
        - Print CNR, step_sigma, and composite_sigma
    If identifier is given, only process scan folders containing the identifier.
    """
    if not Q3_AVAILABLE:
        print("/!\\  Q3_dev_workspace module not available. Cannot perform slice analysis.")
        print("    Ensure Q3_dev_workspace.py is in the AiQuAM_270225 folder.")
        return

    processed_root = settings.get("processed_data_path", "")
    if not processed_root or not os.path.isdir(processed_root):
        print("/!\\  Invalid processed data path.")
        return

    profile = get_beamline_profile(settings.get("beamline", ""))
    param_keys = profile.get("param_keys", {})

    # Find scan folders, optionally filtered by identifier
    scan_ids = sorted([
        d for d in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, d)) and (identifier is None or identifier in d)
    ])
    if not scan_ids:
        print("/!\\  No scan folders found." + (f" (filtered by '{identifier}')" if identifier else ""))
        return

    print("\n--- Slice Analysis Results ---")
    if identifier:
        print(f"Filtering scans by: '{identifier}'")
    print(f"\n{'SCAN_ID':<60} {'SLICE_FILE':<30} {'CNR':>10} {'STEP_σ':>12} {'COMP_σ':>12}")
    print("-" * 124)

    total_slices = 0
    for scan_id in scan_ids:
        scan_path = os.path.join(processed_root, scan_id)
        slice_files = []
        
        # Find all slice.vol files
        for root, dirs, files in os.walk(scan_path):
            for f in files:
                if f == 'slice.vol' or f.endswith('slice.vol'):
                    slice_files.append(os.path.join(root, f))

        if not slice_files:
            continue

        # Process each slice file
        for slice_file in sorted(slice_files):
            total_slices += 1
            slice_name = os.path.basename(slice_file)
            
            # Find corresponding info file for dimensions
            info_file = slice_file + ".info"
            if not os.path.isfile(info_file):
                print(f"{scan_id:<60} {slice_name:<30} {'NO INFO FILE':>10} {'-':>12} {'-':>12}")
                continue

            try:
                # Parse info file for dimensions
                params = parse_text_file(info_file)
                shape_z = int(params.get(param_keys.get("vol_size_z", "NUM_Z")) or params.get("NUM_Z") or params.get("vol_size_z"))
                shape_y = int(params.get(param_keys.get("vol_size_y", "NUM_Y")) or params.get("NUM_Y") or params.get("vol_size_y"))
                shape_x = int(params.get(param_keys.get("vol_size_x", "NUM_X")) or params.get("NUM_X") or params.get("vol_size_x"))
                
                # Load slice data using memory mapping
                dtype = np.float32
                raw = np.memmap(slice_file, dtype=dtype, mode='r')
                
                # Reshape to get the volume (slice files are usually single slices or thin volumes)
                # Try to reshape - if it's a single slice, shape_z might be 1
                expected_size = shape_z * shape_y * shape_x
                if raw.size != expected_size:
                    # Try interpreting as (x, y, z) in Fortran order
                    data = np.reshape(raw, (shape_x, shape_y, shape_z), order='F')
                else:
                    data = np.reshape(raw, (shape_z, shape_y, shape_x))
                
                # Take middle slice if volume, or first slice if single
                if data.shape[0] > 1:
                    vol_slice = data[data.shape[0] // 2, :, :]
                else:
                    vol_slice = data[0, :, :]
                
                # Apply Q3 analysis
                result = Q3_dev_workspace.slice_quick_IQM(
                    vol_slice=vol_slice,
                    label_mask=None,
                    particle_label=None,
                    fit_function=None,
                    margin=10,
                    min_particle_size=10,
                    max_particle_size=10000,
                    roi_margin=25
                )
                
                cnr = result['CNR']
                step_sig = result['step_sigma']
                comp_sig = result['composite_sigma']
                
                # Format output
                cnr_str = f"{cnr:.3f}" if not np.isnan(cnr) else "NaN"
                step_str = f"{step_sig:.3f}" if not np.isnan(step_sig) else "NaN"
                comp_str = f"{comp_sig:.3f}" if not np.isnan(comp_sig) else "NaN"
                
                print(f"{scan_id:<60} {slice_name:<30} {cnr_str:>10} {step_str:>12} {comp_str:>12}")
                
                # Clean up memory
                del raw, data, vol_slice
                
            except Exception as e:
                print(f"{scan_id:<60} {slice_name:<30} {'ERROR':>10} {str(e)[:20]:>12} {'-':>12}")

    print("-" * 124)
    print(f"\nTotal slices analyzed: {total_slices}")
    if total_slices == 0:
        print("/!\\  No slice.vol files found in the selected scans.")
        
# ---------------------------------------------
# Main REPL Loop
# ---------------------------------------------

def main():
    settings = {}
    print("Welcome to the workflow shell. Type 'help' for commands. Ctrl+D to quit.")

    # Initial setup
    settings = init_settings(settings)

    commands = {
        'initialize': lambda: init_settings(settings),
        'show_settings': lambda: show_settings(settings),
        #'list': lambda: list_raw_files(settings),
        #'scan': lambda: scan_parameters(settings),
        'check': lambda: check_reconstruction_status(settings),
        'check_detail': lambda: check_reconstruction_details(settings),
        'pending': lambda: show_pending_reconstructions(settings),
        'pending_detail': lambda: show_pending_details(settings),
        'scan_views': None,  # We'll handle this manually below
        'exit': None
    }

    while True:
        try:
            cmd = input("\n> ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            # Only ask if the user wants to exit, do not re-initialize or reset anything
            try:
                confirm = input("\n/!\\  Do you really want to exit? (y/N): ").strip().lower()
            except EOFError:
                print('\nGoodbye!')
                break
            if confirm == 'y':
                print("\nGoodbye!")
                break
            else:
                continue

        if cmd == 'exit':
            print("Goodbye!")
            break
        elif cmd == 'help':
            print("\nAvailable commands:")
            for k in commands:
                if commands[k] is not None:
                    print(f"  {k}")
            print("  details <identifier>")
            print("  scan_views [identifier]")
            print("  analyze_slices [identifier]")
            print("  help")
            print("  exit")
        elif cmd.startswith('scan_views'):
            parts = cmd.split(maxsplit=1)
            identifier = parts[1] if len(parts) > 1 else None
            create_ortho_slices(settings, identifier)
        elif cmd.startswith('analyze_slices'):
            parts = cmd.split(maxsplit=1)
            identifier = parts[1] if len(parts) > 1 else None
            analyze_slices(settings, identifier)
        elif cmd.startswith('details '):
            identifier = cmd.split(' ', 1)[1].strip()
            if identifier:
                show_scan_details(settings, identifier)
            else:
                print("Please provide an identifier, e.g. 'details X001'")
        elif cmd in commands and commands[cmd]:
            if cmd == 'init':
                settings = commands[cmd]()
            else:
                result = commands[cmd]()
                if cmd == 'check' and result:
                    pass
        else:
            print("Unknown command. Type 'help' to see available commands.")

if __name__ == "__main__":
    main()
