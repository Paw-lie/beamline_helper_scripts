#creates a csv file of the processed data folder showing if reconstructed volumes are in the folders and what format they are in
#designed for the 2.7 Python version available with 'Python' in console.
import os
import csv

def find_reconstructed_folder(path):
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            if d == 'reconstructed_volumes':
                return os.path.join(root, d)
    return None

def scan_folder_structure(root_folder, save_path):
    results = []

    for item in os.listdir(root_folder):
        scan_path = os.path.join(root_folder, item)
        if not os.path.isdir(scan_path):
            continue

        identifier = item  # Always use full folder name

        rec_folder = find_reconstructed_folder(scan_path)

        vol_info = vol_vol = tif = raw_vol = False

        if rec_folder and os.path.isdir(rec_folder):
            for root, _, files in os.walk(rec_folder):
                for file in files:
                    file_lower = file.lower()
                    if file_lower.endswith('.vol.info'):
                        vol_info = True
                    elif file_lower.endswith('vol.vol'):
                        vol_vol = True
                    elif  file_lower.endswith('.raw'):
                        raw_vol = True
                    elif file_lower.endswith('tif'):
                        tif = True
                    elif file_lower.endswith('tiff'):
                        tif = True

        has_volume = vol_vol or tif or raw_vol

        results.append({
            'Scan Identifier': identifier,
            'Has Volume': has_volume,
            'vol.info': vol_info,
            'vol.vol': vol_vol,
            'tif': tif,
            '.raw': raw_vol
        })


    # Write CSV
    with open(save_path, mode='wb') as csvfile:
        fieldnames = ['Scan Identifier', 'Has Volume', 'vol.info', 'vol.vol', 'tif', '.raw']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

# Example usage:
# scan_folder_structure('/your/scan/folder', '/your/output/report.csv')



scan_folder_structure('/gpfs/ga/data/visitor/md1430/id19/20250502/PROCESSED_DATA/', '/home/esrf/paul0307/Desktop/scan_overview_27052025.csv')
