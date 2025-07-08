import os
import csv
from fileIO import tif_to_raw
def find_reconstructed_folder(path):
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            if d == 'reconstructed_volumes':
                return os.path.join(root, d)
    return None

def delete_tiff_files(folder_path):
	for filename in os.listdir(folder_path):
		if filename.lower().endswith(('.tif','.tiff')):
			file_path = os.path.join(folder_path, filename)
			if os.path.isfile(file_path):
				os.remove(file_path)
				print("Deleted:", file_path)

def scan_folder_structure(root_folder, save_path):
    results = []
    for item in os.listdir(root_folder):
        scan_path = os.path.join(root_folder, item)
        if not os.path.isdir(scan_path):
            continue

        identifier = item  # Always use full folder name

        rec_folder = find_reconstructed_folder(scan_path)
        found = False
        vol_info = vol_vol = tif = False
        if rec_folder and os.path.isdir(rec_folder):
            for root, _, files in os.walk(rec_folder):
                for file in files:
                    file_lower = file.lower()
                    if file_lower.endswith('tif') or file_lower.endswith('tiff') and found is False:
                    	tif = True
                    	file_name = file
                    	print(file_name)
                    	tif_to_raw(rec_folder + '/' + file_name, rec_folder + '/')
                    	found = True
                    	delete_tiff_files(rec_folder)
#        if rec_folder is not None:
#	        print(rec_folder + '/')
#	        print(file_name)
                    	#fileIO.tif_to_raw(rec_folder + file, rec_folder + file)

# Example usage:
# scan_folder_structure('/your/scan/folder', '/your/output/report.csv')



scan_folder_structure('/gpfs/ga/data/visitor/md1430/id19/20250502/PROCESSED_DATA/', '/home/esrf/paul0307/Desktop/scan_overview_26052025.csv')
