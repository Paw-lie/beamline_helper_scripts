# Import functions from Q3_open_source_fit_two_half_gaussians.py
from Q3_open_source_fit_two_half_gaussians import (
	plot_erf, erf_function, extract_param_values_from_fit_model, half_gaussian, full_gaussian,
	fit_error_function, plot_error_function_fit, fit_composite_TwoHalfGaussians, fit_composite_ErfWithFlankingGaussians,
	fit_composite_TwoFullGaussians, fit_composite_PREFFERRED_MODEL_ATM_09102024, fit_tmp, fit_composite_SigmoidWithOvershoots,
	plot_two_half_gaussians, create_scanslice_overview_old, create_scanslice_overview, create_particle_overview,
	create_sigma_plot, compute_single_interface_averages, prepare_mask, analyze_slice
)

# Import functions from Q3_open_source_image_processing_functions.py
from Q3_open_source_image_processing_functions import (
	border_point_calculation_open_source, create_other_label_exclusion_mask, get_equally_spaced_integers, get_slice,
	get_orth_slice, show_slice, show, calculate_3d_distance_map, calc_3d_CNR_open_source, erode, dilate,
	connected_component_analysis, filter_and_sort_components, connected_component_3d, get_ROI, fill_holes,
	calculate_otsu_threshold, create_circular_mask, threshold
)

import matplotlib.pyplot as plt  # kept for potential quick plotting (not interactive buttons)
import fileIO as fileIO
import numpy as np
import os
import glob
import json
from pathlib import Path

#a space for me to develop and test code snippets

"""Development script to batch extract pore interface profiles from .raw volumes.
Non-interactive version: scans for .raw files, extracts profiles, saves compact npz results.
"""

# Configuration
SCAN_ROOT = r'z:\46_AIQuAM3D\01_beamtimes'
OUTPUT_DIR = r'TestDatasets'
N = 1  # number of equally spaced slices to analyze (excluding first/last)
PORES_PER_SLICE = 3  # how many pores per slice to keep
MASK_MARGIN = 0  # mask prep margin, keep 0 for robust ROIs
ROI_MARGIN = 25  # extra pixels around each pore bounding box for the ROI
LOCAL_TOMO = False

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all .raw files recursively
raw_files = glob.glob(os.path.join(SCAN_ROOT, '**', '*.raw'), recursive=True)
print(f"Found {len(raw_files)} .raw files in {SCAN_ROOT}")

# Process each file
for file_idx, vol_path in enumerate(raw_files):
    print(f"\n{'='*80}")
    print(f"Processing file {file_idx+1}/{len(raw_files)}: {os.path.basename(vol_path)}")
    print(f"{'='*80}")
    
    try:
        # Try to load the volume
        vol = fileIO.memMapRawVolume(vol_path)
        print(f"Volume shape: {vol.shape}")
        
        # Select slices (exclude first/last to avoid border effects)
        slice_indices = get_equally_spaced_integers(0, vol.shape[0] - 1, N + 2)[1:-1]
        
        # Fetch those slices
        slices = get_slice(vol, slice_indices)
        
        # Compute default min/max particle size based on 0.0001%/0.1% of plane pixels
        plane_pixels = slices[0].shape[0] * slices[0].shape[1]
        min_particle_size = max(1, int(plane_pixels * 0.000001))
        max_particle_size = max(1, int(plane_pixels * 0.001))
        
        # Output collectors for quick fitting/testing
        x_profiles = []
        y_profiles = []
        pore_masks = []
        roi_images = []
        profiles_meta = []
        CNR_list = []
        
        for i, s_idx in enumerate(slice_indices):
            slc = slices[i]
            # Prepare mask like in printIQreport
            mask = prepare_mask(slc, volume_mask=None, mask_path=None, margin=MASK_MARGIN, slice_index=s_idx, local_tomo=LOCAL_TOMO)

            # CNR per slice (for logging/reference)
            # Create two masks: mask_particles (foreground) and mask_background
            mask_particles = mask.astype(bool)  # particles = True
            mask_background = ~mask_particles   # background = True (inverted)
            slice_CNR = calc_3d_CNR_open_source(slc, mask_particles, mask_background)
            CNR_list.append(slice_CNR)
            print(f"  Slice {s_idx} CNR: {slice_CNR:.2f}")

            # Connected components and filtering similar to printIQreport
            labeled_volume, number_of_labels = connected_component_3d(mask)
            label_analysis = connected_component_analysis(labeled_volume, number_of_labels)
            filtered_label_analysis = filter_and_sort_components(
                label_analysis,
                slc.shape,
                top_n=PORES_PER_SLICE,
                min_size=min_particle_size,
                max_size=max_particle_size,
            )

            if not filtered_label_analysis:
                print("  No components passed filtering criteria.")
                continue

            # For each pore, build ROI mask/image and extract x/y profile
            for particle in filtered_label_analysis[:PORES_PER_SLICE]:
                bbox = particle.get('Bounding Box')
                comp_id = particle.get('Component')
                num_px = particle.get('Number of pixels')
                

                roi_mask = get_ROI(labeled_volume, bbox, margin=ROI_MARGIN)
                roi_img = get_ROI(slc, bbox, margin=ROI_MARGIN)
                particle_CNR = calc_3d_CNR_open_source(roi_img, threshold(roi_mask, comp_id, comp_id), threshold(roi_mask, 0, 0))

                x, y = border_point_calculation_open_source(roi_img, roi_mask, label_id=comp_id)
                if x is None or y is None:
                    continue

                x_profiles.append(np.asarray(x))
                y_profiles.append(np.asarray(y))
                pore_masks.append(roi_mask)
                roi_images.append(roi_img)
                profiles_meta.append({
                    'slice_index': int(s_idx),
                    'component_id': int(comp_id) if comp_id is not None else None,
                    'bbox': bbox,
                    'num_pixels': int(num_px) if num_px is not None else None,
                    'whole_slice_CNR': float(slice_CNR),
                    'particle_CNR': float(particle_CNR),
                })
        
        print(f"  Collected {len(x_profiles)} profiles from {len(slice_indices)} slices.")
        
        # Save results in compact format if we got any profiles
        if len(x_profiles) > 0:
            base_name = Path(vol_path).stem
            output_file = os.path.join(OUTPUT_DIR, f"{base_name}_profiles.npz")
            
            # Convert ragged lists to object arrays to avoid shape errors
            x_obj = np.array(x_profiles, dtype=object)
            y_obj = np.array(y_profiles, dtype=object)
            masks_obj = np.array(pore_masks, dtype=object)
            roi_obj = np.array(roi_images, dtype=object)

            try:
                np.savez_compressed(
                    output_file,
                    x_profiles=x_obj,
                    y_profiles=y_obj,
                    pore_masks=masks_obj,
                    roi_images=roi_obj,
                    profiles_meta=json.dumps(profiles_meta),  # Serialize metadata as JSON string
                    CNR_list=np.asarray(CNR_list),
                    volume_shape=np.asarray(vol.shape),
                    slice_indices=np.asarray(slice_indices),
                    file_path=np.asarray(vol_path),
                )
            except Exception as save_err:
                print(f"  ✗ Failed saving compressed npz due to {save_err}; retrying uncompressed")
                np.savez(
                    output_file,
                    x_profiles=x_obj,
                    y_profiles=y_obj,
                    pore_masks=masks_obj,
                    roi_images=roi_obj,
                    profiles_meta=json.dumps(profiles_meta),
                    CNR_list=np.asarray(CNR_list),
                    volume_shape=np.asarray(vol.shape),
                    slice_indices=np.asarray(slice_indices),
                    file_path=np.asarray(vol_path),
                )
            print(f"  ✓ Saved to {output_file}")
        else:
            print(f"  ✗ No profiles found, skipping save.")
            
    except Exception as e:
        print(f"  ✗ Error processing {vol_path}: {e}")
        continue

print(f"\n{'='*80}")
print(f"Processing complete. Results saved to {OUTPUT_DIR}/")
print(f"{'='*80}")