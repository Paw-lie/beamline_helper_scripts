"""
Quick analysis utilities for pore profile extraction and fitting.

Main functions:
- analyze_slice_quick: Analyze a single volume slice
- analyze_npz_slice_quick: Load and analyze a slice from NPZ file
"""

import numpy as np
import json
import Q3_open_source_image_processing_functions as img_proc
import Q3_open_source_fit_two_half_gaussians as function_fitting


def slice_quick_IQM(vol_slice, label_mask=None, particle_label=None, fit_function=None, 
                        margin=10, min_particle_size=10, max_particle_size=1000, roi_margin=25):
    """
    Analyze a single volume slice: calculate CNR, extract profiles, fit, and return results.
    
    Parameters:
    -----------
    vol_slice : numpy.ndarray
        2D slice from volume
    label_mask : numpy.ndarray, optional
        Label mask with labeled particles. If None, will threshold and label automatically.
    particle_label : int, optional
        Specific particle label ID to analyze. If None, uses the largest particle.
    fit_function : callable, optional
        Fit function to use. Defaults to fit_composite_PREFFERRED_MODEL_ATM_09102024.
    margin : int
        Margin for mask preparation
    min_particle_size : int
        Minimum particle size for filtering
    max_particle_size : int
        Maximum particle size for filtering
    roi_margin : int
        Margin around particle bounding box for ROI extraction
        
    Returns:
    --------
    dict
        Dictionary with keys: 'CNR', 'step_sigma', 'composite_sigma', 'x_profile', 'y_profile', 'fit_model'
    """
    # Create label mask if not provided
    if label_mask is None:
        mask = img_proc.prepare_mask(vol_slice, volume_mask=None, mask_path=None, 
                                     margin=margin, slice_index=0, local_tomo=False)
        labeled_volume, _ = img_proc.connected_component_3d(mask)
    else:
        labeled_volume = label_mask
    
    # Calculate CNR for specific particle or all particles
    if particle_label is not None:
        mask_particles = (labeled_volume == particle_label).astype(bool)
        mask_background = (labeled_volume == 0).astype(bool)
    else:
        mask_particles = (labeled_volume > 0).astype(bool)
        mask_background = ~mask_particles
    CNR = img_proc.calc_3d_CNR_open_source(vol_slice, mask_particles, mask_background)
    
    # Get components and filter
    label_analysis = img_proc.connected_component_analysis(labeled_volume)
    
    if particle_label is not None:
        filtered_label_analysis = [comp for comp in label_analysis 
                                  if comp.get('Component') == particle_label]
    else:
        filtered_label_analysis = img_proc.filter_and_sort_components(
            label_analysis, vol_slice.shape, top_n=1, 
            min_size=min_particle_size, max_size=max_particle_size
        )
    
    if not filtered_label_analysis:
        return {
            'CNR': CNR, 'step_sigma': np.nan, 'composite_sigma': np.nan,
            'x_profile': None, 'y_profile': None, 'fit_model': None
        }
    
    # Extract profile from particle
    particle = filtered_label_analysis[0]
    bbox = particle.get('Bounding Box')
    comp_id = particle.get('Component')
    
    roi_mask = img_proc.get_ROI(labeled_volume, bbox, margin=roi_margin)
    roi_img = img_proc.get_ROI(vol_slice, bbox, margin=roi_margin)
    x, y = img_proc.border_point_calculation_open_source(roi_img, roi_mask, label_id=comp_id)
    
    if x is None or y is None:
        return {
            'CNR': CNR, 'step_sigma': np.nan, 'composite_sigma': np.nan,
            'x_profile': None, 'y_profile': None, 'fit_model': None
        }
    
    # Fit profile
    if fit_function is None:
        fit_function = function_fitting.fit_composite_PREFFERRED_MODEL_ATM_09102024
    
    try:
        fit_model = fit_function(np.asarray(x), np.asarray(y))
        step_sigma = fit_model.params.get('step_sigma', type('obj', (), {'value': np.nan})).value
        
        # Calculate composite sigma (weighted average of step + overshoot + undershoot)
        required_params = ['step_sigma', 'step_amplitude', 'overshoot_sigma', 
                          'overshoot_amplitude', 'undershoot_sigma', 'undershoot_amplitude']
        if all(p in fit_model.params for p in required_params):
            step_s = abs(fit_model.params['step_sigma'].value)
            step_a = abs(fit_model.params['step_amplitude'].value)
            over_s = abs(fit_model.params['overshoot_sigma'].value)
            over_a = abs(fit_model.params['overshoot_amplitude'].value)
            under_s = abs(fit_model.params['undershoot_sigma'].value)
            under_a = abs(fit_model.params['undershoot_amplitude'].value)
            composite_sigma = (step_s * step_a + over_s * over_a + under_s * under_a) / (step_a + over_a + under_a)
        else:
            composite_sigma = np.nan
    except Exception:
        fit_model = None
        step_sigma = np.nan
        composite_sigma = np.nan
    
    return {
        'CNR': CNR,
        'step_sigma': step_sigma,
        'composite_sigma': composite_sigma,
        'x_profile': x,
        'y_profile': y,
        'fit_model': fit_model
    }


def slice_quick_IQM_npz(npz_path, slice_index=0, particle_label=None, 
                            fit_function=None, roi_margin=25):
    """
    Load npz file, extract ROI and mask, and apply slice_quick_IQM.
    
    Parameters:
    -----------
    npz_path : str or Path
        Path to the NPZ file
    slice_index : int, optional
        Index of the slice/ROI to analyze (default: 0)
    particle_label : int, optional
        Specific particle label ID to analyze. If None, uses component_id from metadata.
    fit_function : callable, optional
        Fit function to use. Defaults to fit_composite_PREFFERRED_MODEL_ATM_09102024.
    roi_margin : int
        Margin around particle bounding box for ROI extraction
        
    Returns:
    --------
    dict
        Dictionary with keys: 'CNR', 'step_sigma', 'composite_sigma', 
        'x_profile', 'y_profile', 'fit_model', 'metadata'
    """
    # Load npz file
    data = np.load(npz_path, allow_pickle=True)
    roi_images = data['roi_images']
    pore_masks = data['pore_masks']
    profiles_meta = json.loads(str(data['profiles_meta']))
    
    # Get the specified slice/ROI
    if slice_index >= len(roi_images):
        raise IndexError(f"slice_index {slice_index} out of range. File contains {len(roi_images)} ROIs.")
    
    vol_slice = roi_images[slice_index]
    label_mask = pore_masks[slice_index]
    metadata = profiles_meta[slice_index]
    
    # Use component_id from metadata if no particle_label specified
    if particle_label is None:
        particle_label = metadata.get('component_id')
    
    # Apply analyze_slice_quick
    result = slice_quick_IQM(
        vol_slice=vol_slice,
        label_mask=label_mask,
        particle_label=particle_label,
        fit_function=fit_function,
        roi_margin=roi_margin
    )
    
    result['metadata'] = metadata
    return result

