#scans the folders to find which scans have no Processed_data_folder
import os
import csv

def find_missing_scan_folders(f1, f2, save_csv_path):
	s1 = set(os.listdir(f1))
	s2 = set(os.listdir(f2))
	diff = [d for d in s1 -s2 if os.path.isdir(os.path.join(f1,d))]
	with open(save_csv_path, "w") as f:
		csv.writer(f).writerows([[d] for d in diff])
		
raw_folder ='/gpfs/ga/data/visitor/in1187/bm18/20250604/RAW_DATA/'
processed_folder = '/gpfs/ga/data/visitor/in1187/bm18/20250604/PROCESSED_DATA/'
save_path = '/gpfs/ga/data/visitor/in1187/bm18/20250604/SCRIPTS/helper_output/not_yet_reconstructed.csv'

find_missing_scan_folders(raw_folder, processed_folder, save_path)
