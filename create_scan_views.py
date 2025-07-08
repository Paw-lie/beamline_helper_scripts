import os
import os
import numpy as np
import matplotlib.pyplot as plt

def find_reconstructed_folder(path):
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            if d == 'reconstructed_volumes':
                return os.path.join(root, d)
    return None

def parse_vol_info(vol_info_path):
    with open(vol_info_path, 'r') as f:
        lines = f.readlines()[:3]
    try:
        num_x = int(lines[0].split('=')[1].strip())
        num_y = int(lines[1].split('=')[1].strip())
        num_z = int(lines[2].split('=')[1].strip())
        return (num_x, num_y, num_z)
    except:
        raise ValueError("vol.info file is not in the expected format")

def process_scan(scan_path, save_path):
    folder_name = os.path.basename(scan_path)
    rec_folder = find_reconstructed_folder(scan_path)

    if not rec_folder:
        print("No reconstructed_volumes in", scan_path)
        return

    vol_info_file = None
    vol_file = None

    for file in os.listdir(rec_folder):
        file_lower = file.lower()
        if file_lower.endswith('.vol.info'):
            vol_info_file = os.path.join(rec_folder, file)
        elif file_lower.endswith('vol.vol'):
            vol_file = os.path.join(rec_folder, file)

    if not vol_info_file or not vol_file:
        print("Missing vol files in", rec_folder)
        return

    shape = parse_vol_info(vol_info_file)
    dtype = np.float32

    expected_size = shape[0] * shape[1] * shape[2] * 4  # float32 = 4 bytes
    actual_size = os.path.getsize(vol_file)

    if expected_size != actual_size:
        print("Warning: File size mismatch in", folder_name)
        print("Expected:", expected_size, "Actual:", actual_size)
        return

    try:
        raw = np.memmap(vol_file, dtype=dtype, mode='r')
        data = np.reshape(raw, shape, order='F')  # Fortran-order reshape

        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        slice_xy = data[:, :, cz]
        slice_xz = data[:, cy, :]
        slice_yz = data[cx, :, :]

        # Plot orthoslices
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(slice_xy.T, cmap='gray', origin='lower')
        axes[0].set_title('XY (Z=%d)' % cz)
        axes[1].imshow(slice_xz.T, cmap='gray', origin='lower')
        axes[1].set_title('XZ (Y=%d)' % cy)
        axes[2].imshow(slice_yz.T, cmap='gray', origin='lower')
        axes[2].set_title('YZ (X=%d)' % cx)

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out_file = os.path.join(save_path, folder_name + '.png')
        plt.savefig(out_file, dpi=150)
        plt.close()
        print("Saved:", out_file)

    except Exception as e:
        print("Error processing", folder_name + ":", str(e))

def process_all_scans(root_folder, save_path):
    for item in os.listdir(root_folder):
        scan_path = os.path.join(root_folder, item)
        if os.path.isdir(scan_path):
            process_scan(scan_path, save_path)

def process_all_scans(root_folder, save_path):
    for item in os.listdir(root_folder):
        scan_path = os.path.join(root_folder, item)
        if os.path.isdir(scan_path) and 'X035' in scan_path:
            process_scan(scan_path, save_path)

# Example usage:
# process_all_scans('/path/to/scan/root', '/path/to/save/images')

process_all_scans('/gpfs/ga/data/visitor/in1187/bm18/20250604/PROCESSED_DATA/','/home/esrf/paul0307/scripts/beamline_helper_scripts/scan_views/')
