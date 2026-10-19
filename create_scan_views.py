
def create_scan_ortho_view_raw(data_path, save_path, dtype, shape, subsampling=None):
    data = np.memmap(data_path, dtype=dtype, mode='r', shape=shape, order='F')
    #data = np.reshape(raw, shape, order='F')

    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    #slice_xy = data[:, :, cz].copy()
    if subsampling is None:
	    slice_xz = data[:, cy, :].copy()
    else:
        slice_xz = data[::subsampling, cy, ::subsampling].copy()
    
    #slice_yz = data[cx, :, :].copy()

    # Save ortho-slices as .raw files with the same ending policy
    dtype_str = str(np.dtype(dtype).name)
    #xy_name = f"{folder_name}_xy_{dtype_str}_{shape[0]}_{shape[1]}_1.raw"
    #xz_name = f"{folder_name}_xz_{dtype_str}_{slice_xz.shape[0]}_{slice_xz.shape[1]}_1.raw"
    #yz_name = f"{folder_name}_yz_{dtype_str}_{shape[1]}_{shape[2]}_1.raw"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #slice_xy.astype(dtype).tofile(os.path.join(save_path, xy_name))
   	#slice_xz.astype(dtype).tofile(os.path.join(save_path, xz_name))
    #slice_yz.astype(dtype).tofile(os.path.join(save_path, yz_name))
    fileIO.saveRawVolumeIJ(slice_xz, save_path + data_path.split('/')[-1].split('_uin')[0])
    print(f"Saved raw ortho-slices: {save_path + data_path.split('/')[-1].split('_uin')[0]}")
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
    raw_file = None
    raw_shape = None
    raw_dtype = None

    # Helper to map dtype string to numpy dtype
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'uint16': np.uint16,
        'uint8': np.uint8,
        'int16': np.int16,
        'int32': np.int32,
        'int64': np.int64,
    }

    for file in os.listdir(rec_folder):
        file_lower = file.lower()
        if file_lower.endswith('.vol.info'):
            vol_info_file = os.path.join(rec_folder, file)
        elif file_lower.endswith('vol.vol'):
            vol_file = os.path.join(rec_folder, file)
        elif file_lower.endswith('.raw'):
            # Parse dtype and shape from filename
            parts = file[:-4].split('_')
            if len(parts) >= 4:
                raw_dtype_str = parts[-4]
                try:
                    raw_shape = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
                    raw_dtype = dtype_map.get(raw_dtype_str, None)
                    if raw_dtype is not None:
                        raw_file = os.path.join(rec_folder, file)
                except Exception:
                    pass

    # If .raw file exists, use it; else fall back to .vol
    if raw_file and raw_shape and raw_dtype:
        shape = raw_shape
        dtype = raw_dtype
        file_to_load = raw_file
        expected_size = shape[0] * shape[1] * shape[2] * np.dtype(dtype).itemsize
        actual_size = os.path.getsize(file_to_load)
    else:
        if not vol_info_file or not vol_file:
            print("Missing vol files in", rec_folder)
            return
        shape = parse_vol_info(vol_info_file)
        dtype = np.float32
        file_to_load = vol_file
        expected_size = shape[0] * shape[1] * shape[2] * 4  # float32 = 4 bytes
        actual_size = os.path.getsize(file_to_load)

    if expected_size != actual_size:
        print("Warning: File size mismatch in", folder_name)
        print("Expected:", expected_size, "Actual:", actual_size)
        return

    try:
        create_scan_ortho_views(file_to_load, save_path, dtype, shape, folder_name)
    except Exception as e:
        print("Error processing", folder_name + ":", str(e))
def create_scan_ortho_views(data_path, save_path, dtype, shape, folder_name):
    raw = np.memmap(data_path, dtype=dtype, mode='r')
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

#process_all_scans('/gpfs/ga/data/visitor/in1187/bm18/20250604/PROCESSED_DATA/','/home/esrf/paul0307/scripts/beamline_helper_scripts/scan_views/')
