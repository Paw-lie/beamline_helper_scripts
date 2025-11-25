#Apply a low level error function fit on slices of a volume to get an impression of the image quality (rough CNR; rough resolution) 
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt
import lmfit
#from fileIO import fileIO
from fileIO import *
#from Q3_open_source_image_processing_functions import *
from Q3_open_source_image_processing_functions as Q3
#from matplotlib.backends.backend_pdf import PdfPages
import Q3_open_source_fit_two_half_gaussians as IQM
def prepare_mask(single_slice, volume_mask, mask_path, margin, slice_index):
    """Prepare a mask for a slice, using either a provided mask or Otsu's thresholding."""
    if mask_path:
        return get_slice(volume_mask, slice_index)
    otsu_value = calculate_otsu_threshold(single_slice)
    return threshold(single_slice, otsu_value, inverse=True)

def fit_error_function(x,y):
	#set up fitting model lmfit
    from lmfit import Model

    #define composite parts of the fitting model: constant + erf + gaussian + gaussian
    consant_mod = lmfit.models.ConstantModel(prefix='const_')
    step_mod = lmfit.models.StepModel(form='erf', prefix='step_')
    sg_mod = Model(half_gaussian, prefix='overshoot_')
    sg_mod2 = Model(half_gaussian, prefix='undershoot_')
    
    erf_params_list = ['step_center', 'step_sigma', 'step_amplitude', 'const_c']
    
    comb_params_list = erf_params_list + ['overshoot_center', 'overshoot_sigma', 'overshoot_amplitude', 'undershoot_center', 'undershoot_sigma', 'undershoot_amplitude']

	# Initial fit with erf-function (constant-function + step-function)
    fit_func = step_mod + consant_mod
    params = fit_func.make_params(const_c = y[0], step_amplitude=np.max(y)-np.min(y), step_center=0)
    params.add('const_c', value=y[0], vary=False)
    params.add('step_amplitude', value=(y[-1]-y[0]), vary=False)
    params.add('step_sigma', value = 1.0, min=0.1*10**(-100), max = np.inf)
    params.add('step_center', value=0.0, vary=True)

    fit_func = fit_func.fit(y, params, x=x)
    #model_fit_params = extract_param_values_from_fit_model(fit_func, erf_params_list)
	
    return fit_func#model_fit_params

def half_gaussian(x, amplitude, center, sigma, positive=True):#, amplitude, center, width, slope, positive=True):
    #gauss = (amplitude / (np.sqrt(2*np.pi) * width)) * np.exp(-(x-center)**2 / (2*width**2))
    result = []
    for elemx in x:
        if (elemx > center and positive) or (elemx < center and not positive):
            gauss = amplitude * np.exp(-(elemx-center)**2 / (2* sigma**2))
            result.append(gauss)
        else:
            result.append(0.0)
            #return 0
    return result

def IQquickFit(file_path=None, mask_path=None, save_path=None, number_of_slices=1, margin=0, cnr_mask_erosion_value=None, bright_pores=False):
    # Validate input file
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file:\n'{file_path}'\ndoes not exist or could not be accessed.")
    print(f"Analyzing file:\n'{file_path}'")
    
    #Load data
    volume = memMapRawVolume(file_path)
    #if bright_pores is True:
	#    volume = np.invert(volume)
    volume_mask = readRawVolume(mask_path) if mask_path else None

    # Select slices to process
    slice_indexes = get_equally_spaced_integers(0, volume.shape[0] - 1, number_of_slices, exclude_outside=True)

    # Initialize data collection
    segmented_relevant_particles_list = []
    box_plot = []
    df_collection = []
    slice_CNR = []
    slice_init_fit_sigma = []
    
    print(f'\033[1mSLICE\tCNR\tresolution\033[0m')
    for slice_index in slice_indexes:
    	#create slice and slice mask
        single_slice = get_slice(volume, slice_index)
        mask = prepare_mask(single_slice, volume_mask, mask_path, margin, slice_index)
        if bright_pores is True:
        	mask = 1 - mask
        
        #calculate CNR for the slice with the mask created or given (CNR might be worse than actually due to multiple phases being in it with otsu creation)
        CNR = calc_3d_CNR_open_source(single_slice, mask, erosion_value=cnr_mask_erosion_value)
        slice_CNR.append(CNR)
        
        #fit an error function and get the sigma from it for an estimation of resolution
        x, y = border_point_calculation_open_source(single_slice, mask, 1, margin=margin)
        init_fit_model = fit_error_function(x, y)
        init_fit_simga = init_fit_model.params['step_sigma'].value
        slice_init_fit_sigma.append(init_fit_simga)
        #print(f'{slice_index}\t{CNR:.2f}\t{init_fit_simga:.2f}')
    import statistics
    print(f'avg_of{number_of_slices}\t{sum(slice_CNR)/len(slice_CNR):.2f}\t{sum(slice_init_fit_sigma)/len(slice_init_fit_sigma):.2f}')


    
#i expect most of the samples to appear hyperintense, therefore bright_pores=True
#IQquickFit(file_path=None, mask_path=None, save_path=None, number_of_slices=1, margin=0, cnr_mask_erosion_value=None, bright_pores=True)
