#!/usr/local/anaconda/envs/ipsdk31_tf_withpandas/bin/python

#this file is a collection of the image processing functions of Q3 i've initialy solved using PyIPSDK.
#Due to the unavailability at some locations here are the open-source solutions i was able to come up with.
#I'm assuming that the volumes handled are np.arrays
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import lmfit
import scipy.ndimage
import fileIO
#from fileIO import fileIO
#from fileIO.fileIO_dev import *
#import fileIO.fileIO_dev as fileIO 
from Q3_open_source_image_processing_functions import *
from matplotlib.backends.backend_pdf import PdfPages
import os
import re

def plot_erf(center, sigma, amplitude, c, xmin=-25, xmax=25, steps=100, **kwargs):
    return plt.plot(np.linspace(xmin,xmax, steps), erf_function(np.linspace(xmin,xmax, steps),center, sigma, amplitude, c), **kwargs)

#error function
def erf_function(x, center, sigma, amplitude, const):
    '''
    output = erf_function(x, center, sigma, amplitude, const)
    _______________________________________________________________________________________________
    Input:
    - x         ; int, float or array   ; greyscale volume to calculate the CNR for
    - center    ; int, float            ; mask seperating the two phases to calculate the CNR for
    - sigma     ; int, float            ; sigma value of the error function
    - amplitude ; int, float            ; amplitude of the error function
    - const     ; int, float            ; offset of the error function
    _______________________________________________________________________________________________
    Output:
    error_function(x) in the same format as x was given (number or array)
    _______________________________________________________________________________________________
    '''
    return ((scipy.special.erf((x-center)/sigma)+1.0)/2*amplitude)+const


def extract_param_values_from_fit_model(fit_model, param_list):
	'''
	This function takes a lm-model fit_model and it's parameter list and returns a dictionary of the parameters and the corresponding values
	'''
	result = {}
	for param in param_list:
		result[param] = fit_model.params[param].value
	return result

def half_gaussian(x, amplitude, center, sigma, positive=True):
    sigma = np.clip(sigma, 1e-8, None)
    result = []
    for elemx in x:
        if (elemx > center and positive) or (elemx < center and not positive):
            gauss = amplitude * np.exp(-(elemx-center)**2 / (2* sigma**2))
            result.append(gauss)
        else:
            result.append(0.0)
    return result

def full_gaussian(x, amplitude, center, sigma):
    #gauss = (amplitude / (np.sqrt(2*np.pi) * width)) * np.exp(-(x-center)**2 / (2*width**2))
    result = []
    for elemx in x:
        gauss = amplitude * np.exp(-(elemx - center) **2 / (2*sigma**2))
        #gauss = (amplitude / (np.sqrt(2*np.pi) * width)) * np.exp(-(elemx-center)**2 / (2*width**2))
        result.append(gauss)
    return result
    
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

def plot_error_function_fit(erf_fit_model_params, x=None, y=None, xrange=None, xsteps=100):
	#takes in the dictionary from an fit_error_function fit and (optionally) the interface-points (x and y) and returns a matplotlib plot
	if x is not None or y is not None:
		plt.scatter(x,y, label='interface distance intensites')
		x = np.linspace(min(x), max(x), xsteps)
	else:
		print('interface points not given, using range, if not specified using default-range')
		if xrange is not None:
			x = np.linspace(xrange[0], xrange[1], xsteps)
		else:
			x = np.linspace(-10, 10, xsteps)
	plt.plot(x, erf_function(x, erf_fit_model_params.get('step_center'), erf_fit_model_params.get('step_sigma'), erf_fit_model_params.get('step_amplitude'), erf_fit_model_params.get('const_c')))
	plt.xlim(min(x), max(x))
	return plt

def fit_composite_TwoHalfGaussians(x,y, initial_fit_values=None):
	#set up fitting model
	from lmfit import Model
	#functions for fitting:
	step_mod = lmfit.models.StepModel(form='erf', prefix='step_')
	consant_mod = lmfit.models.ConstantModel(prefix='const_')
	sg_mod = Model(half_gaussian, prefix='overshoot_')
	sg_mod2 = Model(half_gaussian, prefix='undershoot_')

	erf_params_list = ['step_center', 'step_sigma', 'step_amplitude', 'const_c']
	comb_params_list = erf_params_list + ['overshoot_center', 'overshoot_sigma', 'overshoot_amplitude', 'undershoot_center', 'undershoot_sigma', 'undershoot_amplitude']

	#initialization of the variables of the composite function with selected boundaries
	pars = lmfit.Parameters()
	pars.add('const_c', value=y[0], min=min(y))#, min=y[0], max=y[-1])#, min=y[0], max=y[1])
	pars += step_mod.guess(y,x=x, center=0.0)
	pars.add('delta', value=-2.0, vary=True)
	pars.add('step_center', value=0.0, min=-np.inf, max=np.inf, vary=False)
	pars.add('step_amplitude', value=abs(y[-1]-y[0]), vary=False)
	pars.add('step_sigma', value = 1.0, min=0.1*10**(-10), max = np.inf)
	pars.add('overshoot_center', value=0.0)#expr='step_center')# value=0.0)
	pars.add('overshoot_width', value=1.0)
	pars.add('overshoot_slope', value=1.0)
	pars.add('positive', value=True, vary=False)
	pars.add('undershoot_center', expr='step_center')# value=0.0)
	pars.add('undershoot_width', value=1.0)
	pars.add('undershoot_slope', value=1.0)
	pars.add('undershoot_positive', value=False, vary=False)
	pars.add('overshoot_positive', value=True, vary=False)
	pars.add('overshoot_positive', value=True, vary=True)
	pars.add('overshoot_amplitude', value=abs(max(y)-y[-1]), min=0.0, max=np.inf)
	#pars.add('overshoot_max_value', value=abs(max(y)-y[-1]), min=0.0, max=np.inf)
	pars.add('overshoot_sigma', value=1.1, min=0.0, max=np.inf)
	pars.add('overshoot_center', expr='step_center')# value=0.0, min=0.0)
	pars.add('overshoot_slope', value=1.0)
	pars.add('undershoot_amplitude', value=-abs(y[0]-min(y)), min=-np.inf, max=0.0)
	#pars.add('undershoot_max_value', value=-abs(y[0]-min(y)), min=-np.inf, max=0.0)
	pars.add('undershoot_sigma', value=1.0, min=0.0, max=np.inf)
	pars.add('undershoot_center', expr='step_center')#value=0.0, min=0.0)
	pars.add('undershoot_slope', value=1.0)


	mod_comb_func =consant_mod + step_mod + sg_mod + sg_mod2
	out_comb_func = mod_comb_func.fit(y, params=pars, x=x, method='least_squares')#, max_nfev=500000000)
	model_fit_params = extract_param_values_from_fit_model(out_comb_func, comb_params_list)

	return out_comb_func#model_fit_params

def full_gaussian(x, center, sigma, amplitude):
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def fit_composite_ErfWithFlankingGaussians(x, y):
    # Define model components
    const_mod = lmfit.models.ConstantModel(prefix='const_')
    step_mod = lmfit.models.StepModel(form='erf', prefix='step_')
    overshoot_mod = lmfit.Model(full_gaussian, prefix='overshoot_')
    undershoot_mod = lmfit.Model(full_gaussian, prefix='undershoot_')

    # Composite model
    model = const_mod + step_mod + overshoot_mod + undershoot_mod

    # Define parameters
    pars = lmfit.Parameters()
    pars.add('const_c', value=np.median(y), min=min(y), max=max(y))

    # Step (erf)
    pars.add('step_center', value=np.median(x), min=min(x), max=max(x))
    pars.add('step_amplitude', value=(max(y) - min(y)) / 2, min=0)
    pars.add('step_sigma', value=(max(x)-min(x))/20, min=1e-6)

    # Gaussians: centers flanking the step
    pars.add('overshoot_center', expr='step_center + 5')
    pars.add('overshoot_sigma', value=1.0, min=0.1)
    pars.add('overshoot_amplitude', value=max(y) - np.median(y), min=0)

    pars.add('undershoot_center', expr='step_center - 5')
    pars.add('undershoot_sigma', value=1.0, min=0.1)
    pars.add('undershoot_amplitude', value=-(np.median(y) - min(y)), max=0)

    # Fit model
    result = model.fit(y, pars, x=x)

    return result
	
def fit_composite_TwoFullGaussians(x,y):
	#set up fitting model
	from lmfit import Model
	#functions for fitting:
	step_mod = lmfit.models.StepModel(form='erf', prefix='step_')
	consant_mod = lmfit.models.ConstantModel(prefix='const_')
	#sg_mod = Model(half_gaussian, prefix='overshoot_')
	#sg_mod2 = Model(half_gaussian, prefix='undershoot_')
	sg_mod = Model(full_gaussian, prefix='overshoot_')
	sg_mod2 = Model(full_gaussian, prefix='undershoot_')

	#initialization of the variables of the composite function with selected boundaries
	pars = lmfit.Parameters()
	pars.add('const_c', value=y[0], min=y[0], max=y[-1])#, min=y[0], max=y[1])
	pars += step_mod.guess(y,x=x, center=0.0)
	pars.add('delta', value=-2.0, vary=True)
	pars.add('step_center', value=0.0, min=-np.inf, max=np.inf, vary=False)
	pars.add('step_amplitude', value=abs(y[-1]-y[0]), vary=False)
	pars.add('step_sigma', value = 1.0, min=0.1*10**(-10), max = np.inf)
	
	pars.add('overshoot_center', value=0.0)#expr='step_center')# value=0.0)
	pars.add('overshoot_width', value=1.0)
	pars.add('overshoot_slope', value=1.0)
	
	pars.add('positive', value=True, vary=False)
	
	pars.add('undershoot_center', expr='step_center')# value=0.0)
	pars.add('undershoot_width', value=1.0)
	pars.add('undershoot_slope', value=1.0)
	pars.add('undershoot_positive', value=False, vary=False)
	pars.add('overshoot_positive', value=True, vary=False)
	pars.add('overshoot_amplitude', value=abs(max(y)-y[-1]), min=abs(max(y)-y[-1]), max=np.inf)
	pars.add('overshoot_sigma', value=1.1, min=0.1*10**(-10), max=np.inf)
	pars.add('overshoot_center', value=pars.get('step_center'))# value=0.0, min=0.0)
	pars.add('overshoot_slope', value=0.0)
	pars.add('undershoot_amplitude', value=abs(y[0]-min(y)), min=0.0, max=np.inf)
	pars.add('undershoot_sigma', value=1.0, min=0.1*10**(-10), max=np.inf)
	pars.add('undershoot_center', expr='step_center')#value=0.0, min=0.0)
	pars.add('undershoot_slope', value=1.0)


	mod_comb_func =consant_mod + step_mod + sg_mod - sg_mod2
	out_comb_func = mod_comb_func.fit(y, params=pars, x=x, method='least_squares')#, max_nfev=500000000)
	
	return out_comb_func#model_fit_params

def fit_composite_PREFFERRED_MODEL_ATM_09102024(x,y, initial_fit_values=None):
        #set up fitting model
        from lmfit import Model
        #functions for fitting:
        step_mod = lmfit.models.StepModel(form='erf', prefix='step_')
        consant_mod = lmfit.models.ConstantModel(prefix='const_')
        sg_mod = Model(half_gaussian, prefix='overshoot_')
        sg_mod2 = Model(half_gaussian, prefix='undershoot_')

        erf_params_list = ['step_center', 'step_sigma', 'step_amplitude', 'const_c']
        comb_params_list = erf_params_list + ['overshoot_center', 'overshoot_sigma', 'overshoot_amplitude', 'undershoot_center', 'undershoot_sigma', 'undershoot_amplitude']

        #initialization of the variables of the composite function with selected boundaries
        pars = lmfit.Parameters()
        pars.add('const_c', value=y[0], min=y[0], max=y[-1])#, min=y[0], max=y[1])
        pars += step_mod.guess(y,x=x, center=0.0)
        pars.add('delta', value=-2.0, vary=True)
        pars.add('step_center', value=0.0, min=-np.inf, max=np.inf, vary=False)
        pars.add('step_amplitude', value=abs(y[-1]-y[0]), vary=False)
        pars.add('step_sigma', value = 1.0, min=0.1*10**(-10), max = np.inf)
        pars.add('overshoot_center', value=0.0)#expr='step_center')# value=0.0)
        pars.add('overshoot_width', value=1.0)
        pars.add('overshoot_slope', value=1.0)
        pars.add('positive', value=True, vary=False)
        pars.add('undershoot_center', expr='step_center')# value=0.0)
        pars.add('undershoot_width', value=1.0)
        pars.add('undershoot_slope', value=1.0)
        pars.add('undershoot_positive', value=False, vary=False)
        pars.add('overshoot_positive', value=True, vary=False)
        pars.add('overshoot_positive', value=True, vary=True)
        pars.add('overshoot_amplitude', value=abs(max(y)-y[-1])+500, min=abs(max(y)-y[-1])+500, max=np.inf) #min = 0.0
        pars.add('overshoot_max_value', value=abs(max(y)-y[-1]), min=0.0, max=np.inf)
        pars.add('overshoot_sigma', value=1.1, min=0.0, max=np.inf)
        pars.add('overshoot_center', expr='step_center')# value=0.0, min=0.0)
        pars.add('overshoot_slope', value=1.0)
        pars.add('undershoot_amplitude', value=-abs(y[0]-min(y)), min=-np.inf, max=0.0)
        pars.add('undershoot_max_value', value=-abs(y[0]-min(y)), min=-np.inf, max=0.0)
        pars.add('undershoot_sigma', value=1.0, min=0.0, max=np.inf)
        pars.add('undershoot_center', expr='step_center')#value=0.0, min=0.0)
        pars.add('undershoot_slope', value=1.0)


        mod_comb_func =consant_mod + step_mod + sg_mod + sg_mod2
        out_comb_func = mod_comb_func.fit(y, params=pars, x=x, method='least_squares')#, max_nfev=500000000)
        model_fit_params = extract_param_values_from_fit_model(out_comb_func, comb_params_list)

        return out_comb_func#model_fit_params

def fit_tmp(x,y, initial_fit_values=None):
        #set up fitting model
        from lmfit import Model
        #functions for fitting:
        step_mod = lmfit.models.StepModel(form='erf', prefix='step_')
        consant_mod = lmfit.models.ConstantModel(prefix='const_')
        sg_mod = Model(half_gaussian, prefix='overshoot_')
        sg_mod2 = Model(half_gaussian, prefix='undershoot_')

        erf_params_list = ['step_center', 'step_sigma', 'step_amplitude', 'const_c']
        comb_params_list = erf_params_list + ['overshoot_center', 'overshoot_sigma', 'overshoot_amplitude', 'undershoot_center', 'undershoot_sigma', 'undershoot_amplitude']

        #initialization of the variables of the composite function with selected boundaries
        pars = lmfit.Parameters()
        pars.add('const_c', value=y[0], min=y[0], max=y[-1])#, min=y[0], max=y[1])
        pars += step_mod.guess(y,x=x, center=0.0)
        #pars.add('delta', value=-2.0, vary=True)
        pars.add('step_center', value=0.0, min=-np.inf, max=np.inf, vary=True)
        pars.add('step_amplitude', value=abs(y[-1]-y[0]), vary=True)
        pars.add('step_sigma', value = 1.0, min=10**(-10), max = np.inf)
        pars.add('positive', value=True, vary=False)

        pars.add('overshoot_amplitude', value=abs(max(y)-y[-1]), min=abs(max(y)-y[-1]), max=np.inf) #min = 0.0
        pars.add('overshoot_center', value=x[np.argmax(y)])#expr='step_center')# value=0.0, min=0.0)
        pars.add('overshoot_positive', value=True, vary=False)
        pars.add('overshoot_sigma', value=1.1, min=0.0, max=np.inf)
        pars.add('overshoot_slope', value=1.0)
        pars.add('overshoot_width', value=1.0)

        pars.add('undershoot_amplitude', value=-abs(y[0]-min(y)), min=-np.inf, max=0.0)
        pars.add('undershoot_center', value=x[np.argmin(y)])# expr='step_center')#value=0.0, min=0.0)
        pars.add('undershoot_positive', value=False, vary=False)
        pars.add('undershoot_sigma', value=1.0, min=0.0, max=np.inf)
        pars.add('undershoot_slope', value=1.0)
        pars.add('undershoot_width', value=1.0)
        
        mod_comb_func =consant_mod + step_mod + sg_mod + sg_mod2
        out_comb_func = mod_comb_func.fit(y, params=pars, x=x, method='least_squares')#, max_nfev=500000000)
        model_fit_params = extract_param_values_from_fit_model(out_comb_func, comb_params_list)

        return out_comb_func#model_fit_params

def fit_composite_SigmoidWithOvershoots(x, y):
    """
    Fit a composite model using a sigmoid function with over- and undershoots.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.

    Returns:
        lmfit.model.ModelResult: The result of the fit, including best-fit parameters.
    """
    from lmfit import Model

    def sigmoid_with_overshoots(x, center, step_sigma, amplitude, offset,
                                overshoot_amplitude, overshoot_sigma,
                                undershoot_amplitude, undershoot_sigma):
        """
        Composite model: sigmoid + overshoot (half-Gaussian) + undershoot (half-Gaussian).
        """
        # Sigmoid function
        sigmoid = offset + amplitude / (1 + np.exp(-step_sigma * (x - center)))

        # Overshoot (right side of the transition)
        overshoot = np.where(
            x > center,
            overshoot_amplitude * np.exp(-(x - center) ** 2 / (2 * overshoot_sigma ** 2)),
            0
        )

        # Undershoot (left side of the transition)
        undershoot = np.where(
            x < center,
            undershoot_amplitude * np.exp(-(x - center) ** 2 / (2 * undershoot_sigma ** 2)),
            0
        )

        return sigmoid + overshoot + undershoot

    # Define the composite model
    composite_model = Model(sigmoid_with_overshoots)

    # Initialize parameters for the composite model
    params = composite_model.make_params(
        center=0.0,  # Initial guess for the center of the sigmoid
        step_sigma=1.0,   # Initial guess for the slope of the sigmoid (renamed from 'slope')
        amplitude=max(y) - min(y),  # Amplitude based on data range
        offset=min(y),  # Offset based on the minimum y value
        overshoot_amplitude=abs(max(y) - np.mean(y)) * 0.2,  # Initial guess for overshoot amplitude
        overshoot_sigma=5.0,  # Initial guess for overshoot width
        undershoot_amplitude=-abs(min(y) - np.mean(y)) * 0.2,  # Initial guess for undershoot amplitude
        undershoot_sigma=5.0  # Initial guess for undershoot width
    )

    # Set parameter bounds
    params['center'].set(min=min(x), max=max(x))
    params['step_sigma'].set(min=0.01, max=10.0)  # Renamed from 'slope'
    params['amplitude'].set(min=0.0)
    params['offset'].set(min=0.0)
    params['overshoot_amplitude'].set(min=0.0)
    params['overshoot_sigma'].set(min=0.1)
    params['undershoot_amplitude'].set(max=0.0)
    params['undershoot_sigma'].set(min=0.1)

    # Perform the fit
    result = composite_model.fit(y, params, x=x)

    return result

def plot_two_half_gaussians(comp_fit_model_params, x, y):
	#takes in the lmfit_model from an fit_two_half_gaussians fit and the interface-points (x and y) and returns a matplotlib plot
	plt.scatter(x,y)
	plt.plot(x, comp_fit_model_params.best_fit, label='best comp fit')
	plt.xlim(min(x), max(x))
	return plt

def create_scanslice_overview_old(single_slice, mask, label_mask, x, y, sample_ID=None, slice_number=None):
	#takes in three image objects and something that can be scatter-plotted and returns a plot of the three images
	fig, axs = plt.subplots(2,2, figsize=(10, 10))

	axs[0, 0].imshow(single_slice, cmap='Greys')
	axs[0, 0].set_title('Slice')

	axs[0, 1].imshow(mask, cmap='Greys')
	axs[0, 1].set_title('Mask')

	axs[1, 0].imshow(label_mask, cmap='turbo')
	axs[1, 0].set_title('Connected Component')

	axs[1, 1].scatter(x,y)
	axs[1, 1].set_title('Interface Intensity Profile')
	axs[1, 1].set_box_aspect(1)#ensures that the scatter plot is square

	# Add the main title
	fig.suptitle("Scan Slice Overview", fontsize=16, fontweight='bold')

	# Add a subtitle if sample_ID or slice_number is provided
	if sample_ID is not None or slice_number is not None:
		subtitle = ""
		if sample_ID is not None:
			subtitle += f"Sample ID: {sample_ID}"
		if slice_number is not None:
			if sample_ID is not None:
				subtitle += "\n"
			subtitle += f"Slice Number: {slice_number}"
		fig.text(0.5, 0.92, subtitle, ha='center', fontsize=10, color='gray')  # Add as a subtitle

	plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for the suptitle and subtitle
	#plt.show()
	return plt

def create_scanslice_overview(single_slice, mask, label_mask, x, y, init_fit_model, comb_fit_model, sample_ID=None, slice_number=None):
	#takes in three image objects and something that can be scatter-plotted and returns a plot of the three images
	fig, axs = plt.subplots(2,2, figsize=(10, 10))

	axs[0, 0].imshow(single_slice, cmap='Greys')
	axs[0, 0].set_title('Slice')

	axs[0, 1].imshow(mask, cmap='Greys')
	axs[0, 1].set_title('Mask')

	axs[1, 0].imshow(label_mask, cmap='turbo')
	axs[1, 0].set_title('Connected Component')

	axs[1, 1].scatter(x,y)
	axs[1, 1].plot(x, init_fit_model.best_fit, label='')
	axs[1, 1].plot(x, comb_fit_model.best_fit)
	axs[1, 1].set_title('All Slice-Interface Profile')
	axs[1, 1].set_box_aspect(1)#ensures that the scatter plot is square

	# Add the main title
	fig.suptitle("Scan Slice Overview", fontsize=16, fontweight='bold')

	# Add a subtitle if sample_ID or slice_number is provided
	if sample_ID is not None or slice_number is not None:
		subtitle = ""
		if sample_ID is not None:
			subtitle += f"Sample ID: {sample_ID}"
		if slice_number is not None:
			if sample_ID is not None:
				subtitle += "\n"
			subtitle += f"Slice Number: {slice_number}"
		fig.text(0.5, 0.92, subtitle, ha='center', fontsize=10, color='gray')  # Add as a subtitle

	plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for the suptitle and subtitle
	#plt.show()
	return plt

def create_particle_overview(pore_slice, pore_mask, x, y, init_fit_model, combined_fit_model,sample_ID=None, slice_number=None, pore_number=None, pore_size=None):
	"""
	Creates a composite plot for particle overview.

	Parameters:
	- pore_slice: 2D array representing the pore slice.
	- pore_mask: 2D array representing the pore mask.
	- x, y: Data for the sigma plot.
	- init_fit_model, combined_fit_model: Models for sigma plotting.

	Returns:
	- A matplotlib plt object for further customization or saving.
	"""
	fig, axs = plt.subplots(2, 2, figsize=(10, 10))

	# First row of plots
	axs[0, 0].imshow(pore_slice, cmap='gray')#'Greys')
	axs[0, 0].set_title('Pore Slice')

	axs[0, 1].imshow(pore_mask, cmap='gray')#'turbo')
	axs[0, 1].set_title('Pore Mask')
 
	axs[1, 0].scatter(x,y)
	axs[1, 0].plot(x, combined_fit_model.best_fit, label='best combination fit')
	axs[1, 0].set_xlim(min(x), max(x))
	axs[1, 0].set_title('Combination fit')
	#axs[1, 0].legend()#loc='lower center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
 
	RESOLUTION_AT_UNIT = 1
	RESOLUTION_UNIT = '[px]'
	tmpx = np.linspace(min(x)*RESOLUTION_AT_UNIT,max(x)*RESOLUTION_AT_UNIT,1000)
	fit_func = init_fit_model
	tmp_sigma = fit_func.params['step_sigma'].value
	axs[1, 1].plot(tmpx, #plot the gaussian of the erf of the initial fitted function
				full_gaussian(tmpx,
							1.0, 
							0.0, 
							fit_func.params['step_sigma'].value
							)
				,label=f'initial fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
				)

	tmp_sigma = combined_fit_model.params['step_sigma'].value
	axs[1, 1].fill_between(tmpx, #plot the gaussian of the erf of the fitted combined function
				full_gaussian(tmpx,
							1.0, 
							0.0, 
							combined_fit_model.params['step_sigma'].value
							)
						,alpha = 0.25
						,label=f'error function of the combination fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
				)
	tmp_sigma = combined_fit_model.params['overshoot_sigma'].value
	axs[1, 1].fill_between(tmpx[tmpx>0.0], #plot the gaussian of the overshoot of the fitted combined function
				half_gaussian(tmpx[tmpx>0.0],
							1.0, 
							0.0, 
							combined_fit_model.params['overshoot_sigma'].value
							)
						,alpha = 0.25
						,label=f'overshoot half-gaussian of the combination fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
				)

	tmp_sigma = combined_fit_model.params['undershoot_sigma'].value
	axs[1, 1].fill_between(tmpx[tmpx<0.0], #plot the gaussian of the undershoot of the fitted combined function
				half_gaussian(tmpx[tmpx<0.0],
							1.0, 
							0.0, 
							abs(combined_fit_model.params['undershoot_sigma'].value),
							False
							)
						,alpha = 0.25
						,label=f'undershoot half-gaussian of the combination fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
				)
	axs[1, 1].set_title('Gaussplot')
	axs[1, 1].set_xlabel(f'distance from surface [{RESOLUTION_UNIT}]')
	axs[1, 1].set_xlim((min(x),max(x)))
	axs[1, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
 
 	# Add the main title
	fig.suptitle("Single Pore Analysis", fontsize=16, fontweight='bold')

	# Add a subtitle if sample_ID or slice_number is not provided
	if sample_ID is not None or slice_number is not None:
		subtitle = ""
		if sample_ID is not None:
			subtitle += f"Sample ID: {sample_ID}"
		if slice_number is not None:
			if sample_ID is not None:
				subtitle += "\n"
			subtitle += f"Slice Number: {slice_number}"
		if pore_number is not None:
			if sample_ID is not None:
				subtitle += "\n"
			subtitle += f"Pore Number: {pore_number}"
		if pore_size is not None:
			subtitle += "\n"
			subtitle += f"Pore-size: {pore_size}[px]"
		fig.text(0.5, 0.9, subtitle, ha='center', fontsize=10, color='gray')  # Add as a subtitle

	plt.tight_layout(rect=[0, 0, 1, 0.85])
	#plt.tight_layout()
	return plt

def create_sigma_plot(x, fit_func, out_comb_func, sample_ID=None, particle_ID=None, slice_ID=None, RESOLUTION_AT_UNIT=1, RESOLUTION_UNIT='px'):
    
	#create the overview plot of the sigmas of the different fits:
	tmpx = np.linspace(min(x)*RESOLUTION_AT_UNIT,max(x)*RESOLUTION_AT_UNIT,1000) #create more points to plot at for smoother lineplots
	tmp_sigma = fit_func.params['step_sigma'].value
	title_parts = []
	if sample_ID:
		title_parts.append(f'Sample: {sample_ID}')
	if particle_ID is not None:
		title_parts.append(f'Particle {int(particle_ID)}')
	if slice_ID is not None:
		title_parts.append(f'at Slice {int(slice_ID)}')
	if sample_ID or particle_ID or slice_ID:
		title_parts = ' '.join(title_parts)
		plt.title(f'Gauss-plot of {title_parts}')
	else:
		plt.title('Gauss-plot')

	plt.plot(tmpx, #plot the gaussian of the erf of the initial fitted function
	         full_gaussian(tmpx,
	                       1.0, 
	                       0.0, 
	                       fit_func.params['step_sigma'].value
	                      )
	         ,label=f'initial fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
	         )

	tmp_sigma = out_comb_func.params['step_sigma'].value
	plt.fill_between(tmpx, #plot the gaussian of the erf of the fitted combined function
	         full_gaussian(tmpx,
	                       1.0, 
	                       0.0, 
	                       out_comb_func.params['step_sigma'].value
	                       )
	                 ,alpha = 0.25
	                 ,label=f'error function of the combination fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
	         )
	tmp_sigma = out_comb_func.params['overshoot_sigma'].value
	plt.fill_between(tmpx[tmpx>0.0], #plot the gaussian of the overshoot of the fitted combined function
	         half_gaussian(tmpx[tmpx>0.0],
	                       1.0, 
	                       0.0, 
	                       out_comb_func.params['overshoot_sigma'].value
	                       )
	                 ,alpha = 0.25
	                 ,label=f'overshoot half-gaussian of the combination fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
	         )

	tmp_sigma = out_comb_func.params['undershoot_sigma'].value
	plt.fill_between(tmpx[tmpx<0.0], #plot the gaussian of the undershoot of the fitted combined function
	         half_gaussian(tmpx[tmpx<0.0],
	                       1.0, 
	                       0.0, 
	                       abs(out_comb_func.params['undershoot_sigma'].value),
	                       False
	                       )
	                 ,alpha = 0.25
	                 ,label=f'undershoot half-gaussian of the combination fit (sigma={tmp_sigma:.2f}[{RESOLUTION_UNIT}])'
	         )

	plt.xlabel(f'distance from surface [{RESOLUTION_UNIT}]')
	plt.xlim((min(x),max(x)))
	plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), fancybox=True)
	#plt.subplots_adjust(bottom=0.3)
	return plt

def compute_single_interface_averages(list_x, list_y):
	from collections import defaultdict
	y_values_for_x = defaultdict(list)

	# Loop through each pair of arrays from list_x and list_y
	for x_array, y_array in zip(list_x, list_y):
		for x_val, y_val in zip(x_array, y_array):
			y_values_for_x[x_val].append(y_val)

	# Calculate the average y for each unique x
	average_x = sorted(y_values_for_x.keys())  # Sort x-values for ordered plot
	average_y = []

	for x_val in average_x:
		y_vals = y_values_for_x[x_val]
		if y_vals:  # Only compute mean if list is not empty
			average_y.append(np.nanmedian(y_vals))
		else:
			average_y.append(np.nan)  # Append NaN for empty slices, if needed

	return np.array(average_x), np.array(average_y)


def do_IQ_slicewise_analysis(file_path, mask_path=None, number_of_slices=1, slice_number=None, margin=10, min_particle_size=10, max_particle_size=1000, Pores_to_analyze=3, cnr_mask_erosion_value=None, bright_pores=False, other_label_exclusion_margin=4, local_tomo=False):
    """
    Perform IQ analysis on a given file and mask slice-wise.

    Parameters:
    - file_path: Path to the input file.
    - mask_path: Path to the mask file (optional).
    - number_of_slices: Number of slices to analyze.
    - margin: Margin for analysis.
    - min_particle_size: Minimum particle size for analysis.
    - max_particle_size: Maximum particle size for analysis.
    - Pores_to_analyze: Number of pores to analyze.
    - cnr_mask_erosion_value: Erosion value for CNR mask (optional).
    - bright_pores: Boolean indicating if bright pores are to be analyzed.
    - other_label_exclusion_margin: Exclusion margin for other labels.
    - local_tomo: Boolean indicating if local tomography is used.

    Returns:
    - combined_df: DataFrame containing the analysis results.
    - slice_CNR: List of CNR values for each slice.
    - slice_intensity_profiles: List of intensity profiles for each slice.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file:\n'{file_path}'\ndoes not exist or could not be accessed.")
    print(f"Analyzing file:\n'{file_path}'")
    # Memmap volume (and optionally a mask) for analysis:
    volume = fileIO.memMapRawVolume(file_path)
    volume_mask = fileIO.readRawVolume(mask_path) if mask_path else None
    # Select slices to process
    if slice_number is None:
        slice_indexes = get_equally_spaced_integers(0, volume.shape[0] - 1, number_of_slices, exclude_outside=True)
    else:
        print(f'Specific slice of scan requested. Analyzing single slice number {slice_number}')
        slice_indexes = [slice_number]
    # Set the Model that shall be fit!
    fit_model = fit_composite_ErfWithFlankingGaussians#fit_composite_TwoHalfGaussians#fit_composite_TwoFullGaussians#fit_composite_SigmoidWithOvershoots
    # Initialize data collection
    volume_views = []
    df_collection = []
    slice_CNR = []
    slice_intensity_profiles = []
    # Analyze slices:
    for slice_index in slice_indexes:
        volume_views.append(volume[slice_index,:,:])
        single_slice = get_slice(volume, slice_index)
        mask = prepare_mask(single_slice, volume_mask, mask_path, margin, slice_index, local_tomo=local_tomo)
        CNR = calc_3d_CNR_open_source(single_slice, mask, erosion_value=cnr_mask_erosion_value)
        #print(f'CNR at slice {slice_index}: {CNR:.2f}')
        # Analyze slice
        x, y, labeled_volume, filtered_label_analysis = analyze_slice(
            single_slice, mask, fit_model, slice_index, file_name=None, margin=margin,
            min_particle_size=min_particle_size, max_particle_size=max_particle_size,
            Pores_to_analyze=Pores_to_analyze, pdf=None,
            other_label_exclusion_margin=other_label_exclusion_margin
        )
        # Handle individual particle data
        if Pores_to_analyze > 0:
            particle_data = analyze_particles(
                filtered_label_analysis, labeled_volume, single_slice,
                margin=margin, fit_model=fit_model,
                slice_index=slice_index, file_name=None,
                pdf=None, other_label_exclusion_margin=other_label_exclusion_margin
            )
            df_collection.append(pd.DataFrame(particle_data))
        slice_CNR.append(CNR)
        slice_intensity_profiles.append((x, y))
    # Combine results and filter DataFramei
    # Combine results and filter DataFramei
    if len(df_collection) != 0:
        combined_df = pd.concat(df_collection, ignore_index=True)
    else: 
        combined_df = pd.DataFrame()
    return volume_views, combined_df, slice_CNR, slice_intensity_profiles
    
def printIQreport(
    file_path=None, mask_path=None, save_path=None, number_of_slices=1, margin=10,
    min_particle_size=None, max_particle_size=None, Pores_to_analyze=3, cnr_mask_erosion_value=None,
    bright_pores=False, other_label_exclusion_margin=4, local_tomo=False, skip_empty_report=True
):
    import matplotlib.pyplot as plt
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file:\n'{file_path}'\ndoes not exist or could not be accessed.")
    print(f"Analyzing file:\n'{file_path}'")

    file_name = re.split(r'(_uint8_|_float32_|_uint16_)', os.path.basename(file_path))[0]
    pdf_file_path = os.path.join(save_path, f'IQ_report_Sample_{file_name}.pdf')
    volume = fileIO.memMapRawVolume(file_path)
    if bright_pores is True:
        volume = np.invert(volume)
    volume_mask = fileIO.readRawVolume(mask_path) if mask_path else None

    # --- Set min/max particle size if not given ---
    print(volume.shape[0], volume.shape[1], volume.shape[2])
    plane_pixels = volume.shape[1] * volume.shape[2]
    if min_particle_size is None:
        min_particle_size = max(1, int(plane_pixels * 0.0025))
    if max_particle_size is None:
        max_particle_size = max(1, int(plane_pixels * 0.10))

    # Select slices to process
    slice_indexes = get_equally_spaced_integers(0, volume.shape[0] - 1, number_of_slices, exclude_outside=True)
    fit_model = fit_composite_TwoHalfGaussians

    segmented_relevant_particles_list = []
    box_plot = []
    df_collection = []
    slice_CNR = []

    with PdfPages(pdf_file_path) as pdf:
        for slice_index in slice_indexes:
            single_slice = get_slice(volume, slice_index)
            mask = prepare_mask(single_slice, volume_mask, mask_path, margin, slice_index)
            CNR = calc_3d_CNR_open_source(single_slice, mask, erosion_value=cnr_mask_erosion_value)
            print(f'CNR at slice {slice_index}: {CNR:.2f}')

            x, y, labeled_volume, filtered_label_analysis = analyze_slice(
                single_slice, mask, fit_model, slice_index, file_name, margin,
                min_particle_size, max_particle_size, Pores_to_analyze, pdf, other_label_exclusion_margin
            )

            segmented_relevant_particles_list.append(len(filtered_label_analysis))
            box_plot.append([entry['Number of pixels'] for entry in connected_component_analysis(labeled_volume)])

            particle_data = analyze_particles(
                filtered_label_analysis, labeled_volume, single_slice, margin, fit_model,
                slice_index, file_name, pdf, other_label_exclusion_margin
            )
            df_collection.append(pd.DataFrame(particle_data))
            slice_CNR.append(CNR)

        #ToDo: add boxplot to pdf report again
        #generate_boxplot(box_plot, slice_indexes, number_of_slices, pdf)

    if len(df_collection) != 0:
        combined_df = pd.concat(df_collection, ignore_index=True)
        print(combined_df)
        if not combined_df.empty:
            generate_final_report(combined_df, slice_CNR, save_path, file_name)
        else:
            if skip_empty_report:
                print("No particles found in any slice. No report generated.")
            else:
                raise ValueError("No particles found in any slice. DataFrame is empty.")
    else:
        if skip_empty_report:
            print("No particles found in any slice. No report generated.")
        else:
            raise ValueError("No particles found in any slice. DataFrame is empty.")

    print("Processing complete. Reports and figures saved successfully.")

def plot_views_grid(views, titles=None, cmap='gray'):
    import matplotlib.pyplot as plt
    import math
    plt.cla()
    plt.clf()
    plt.close()
    n = len(views)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(views[i], cmap=cmap)
            if titles:
                ax.set_title(titles[i])
        ax.axis('off')  # Hide axis in all cases

    plt.tight_layout()
    plt.show(block=False)

def showVolOverview(file_path, center_coord=None):
    """
    Show a volume overview of the given file.

    Parameters:
    - file_path: Path to the input file.
    - center_coord: Optional center coordinates for the volume view.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file:\n'{file_path}'\ndoes not exist or could not be accessed.")
    
    # Load volume
    volume = fileIO.memMapRawVolume(file_path)
    
    # Show volume overview
    if center_coord is None:
        center_coord = (volume.shape[0] // 2, volume.shape[1] // 2, volume.shape[2] // 2)
    slice_views = []
    for axis in ['x', 'y', 'z']:
        if axis == 'x':
            slice_view = volume[center_coord[0], :, :]
        elif axis == 'y':
            slice_view = volume[:, center_coord[1], :]
        elif axis == 'z':
            slice_view = volume[:, :, center_coord[2]]
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
        slice_views.append(slice_view)
    plot_views_grid(slice_views, titles=['View1', 'View2', 'View3'])
        
        
    
def show_slice(volume, slice_index, axis='z'):
    """
    Show a specific slice of the volume.

    Parameters:
    - volume: 3D numpy array representing the volume.
    - slice_index: Index of the slice to display.
    - axis: Axis along which to display the slice ('x', 'y', or 'z').
    """
    import matplotlib.pyplot as plt
    if axis == 'x':
        slice_view = volume[slice_index, :, :]
    elif axis == 'y':
        slice_view = volume[:, slice_index, :]
    elif axis == 'z':
        slice_view = volume[:, :, slice_index]
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    return slice_view

    

def consoleIQreport(file_path, *args, **kwargs):
    volume_views ,combined_df, slice_CNR, slice_intensity_profiles = do_IQ_slicewise_analysis(file_path, *args, **kwargs)    # Print results to console.
    file_shape = fileIO.getVolumeParamsFromFilename(file_path)
    slice_indexes = get_equally_spaced_integers(0, file_shape[0][2] - 1, kwargs['number_of_slices'], exclude_outside=True)
    
    plot_views_grid(volume_views, titles=[f'Slice {i}' for i in slice_indexes])
    import plotext as plt
    plt.clf()
    plt.scatter(slice_indexes, slice_CNR)
    plt.ylim(0, min(max(slice_CNR),10))#restrict the bar plot for the CNR from 0 to 10
    plt.title('CNR per slice')
    plt.xlabel('Slice Number')
    plt.ylabel('CNR')
    plt.show()
    plt.clf()
    
    plt.title(f'Slice Intensity Profile')
    plt.xlabel('Distance from Surface')
    plt.ylabel('Intensity')
    for i in range(len(slice_intensity_profiles)):
        x, y = slice_intensity_profiles[i]
        plt.plot(x, y, label=f'Slice {i}')
    plt.show()
    #combined_df, _, _ = do_IQ_slicewise_analysis(file_path, Pores_to_analyze=3, number_of_slices=1)
    return combined_df
    
# Helper functions
def prepare_mask(single_slice, volume_mask, mask_path, margin, slice_index, local_tomo=False):
    """Prepare a mask for a slice, using either a provided mask or Otsu's thresholding."""
    #create a circular mask at the center of the scan if the scan is marked as a local tomography
    if local_tomo is True:
        local_tomo_mask = create_circular_mask(single_slice, reduction_value=5)
    if mask_path:
        return get_slice(volume_mask, slice_index)*local_tomo_mask if local_tomo is True else get_slice(volume_mask, slice_index)
    #median filter the slice before creating the mask to eliminate noise
    single_slice_filtered = scipy.ndimage.median_filter(single_slice, size=5)
    otsu_value = calculate_otsu_threshold(single_slice_filtered)
    mask = threshold(single_slice_filtered, otsu_value, inverse=True)*local_tomo_mask if local_tomo is True else threshold(single_slice_filtered, otsu_value, inverse=True)
    # Apply morphological operations to clean up the mask
    #mask = binary_fill_holes(mask)
    mask = binary_dilation(mask, iterations=10)
    mask = binary_erosion(mask, iterations=10)
    return mask

def analyze_slice(single_slice, mask, fit_model, slice_index, file_name, margin, min_particle_size, max_particle_size, Pores_to_analyze, pdf, other_label_exclusion_margin):
    """Perform analysis on a single slice and save results to the PDF."""
    # Fit the model and calculate border points
    x, y = border_point_calculation_open_source(single_slice, mask, 1, other_label_exclusion_margin=other_label_exclusion_margin)
    if len(x) == 0:  # No valid points
        return x, y, None, []
    
    init_fit_model = fit_error_function(x, y)
    combined_fit_model = fit_model(x, y)
    
    # Plot slice overview and sigma plot
    labeled_volume, number_of_labels = connected_component_3d(mask)
    create_scanslice_overview(single_slice, mask, labeled_volume, x, y, init_fit_model, combined_fit_model, file_name, slice_index)
    if pdf is not None:
        pdf.savefig(bbox_inches='tight')

    #create_sigma_plot(x, init_fit_model, combined_fit_model, sample_ID=file_name, slice_ID=slice_index)
    #pdf.savefig(bbox_inches='tight')

    # Filter connected components
    if pdf is None and Pores_to_analyze == 0:
        filtered_label_analysis = None
    else:
        label_analysis = connected_component_analysis(labeled_volume, number_of_labels)
        filtered_label_analysis = filter_and_sort_components(
            label_analysis, single_slice.shape, top_n=Pores_to_analyze, 
            max_size=max_particle_size, min_size=min_particle_size
        )
        print(f"SLICE {slice_index} | Total Components: {number_of_labels} | Filtered: {len(filtered_label_analysis)}")
    plt.close()
    if pdf is not None:
        # After the function generates figures
        for fig_num in plt.get_fignums():  # Get all figure numbers
            fig = plt.figure(fig_num)      # Get the figure object
            pdf.savefig(fig)               # Save it to PDF
            plt.close(fig)                 # Close the figure

    return x, y, labeled_volume, filtered_label_analysis

def analyze_particles(filtered_label_analysis, labeled_volume, single_slice, margin, fit_model, slice_index, file_name, pdf, CNR_mask_eriosion_value=None, other_label_exclusion_margin=3):
    """Analyze individual particles within a slice."""
    particle_data = []
    for i, particle in enumerate(filtered_label_analysis):
        roi_mask = get_ROI(labeled_volume, particle.get('Bounding Box'), margin=margin)
        roi = get_ROI(single_slice, particle.get('Bounding Box'), margin=margin)
        pore_CNR = calc_3d_CNR_open_source(roi, roi_mask, erosion_value=CNR_mask_eriosion_value)
        #x, y = border_point_calculation_open_source(roi, threshold(roi_mask,i,i, inverse=True), particle.get('Component'), margin=margin)
        x, y = border_point_calculation_open_source(roi, roi_mask, particle.get('Component'), other_label_exclusion_margin=other_label_exclusion_margin)
        init_fit_model = fit_error_function(x, y)
        combined_fit_model = fit_model(x, y)
        if pdf is not None:
            create_particle_overview(
                roi, threshold(roi_mask, particle.get('Component'), particle.get('Component')), x, y, init_fit_model, combined_fit_model,
                sample_ID=file_name, slice_number=slice_index, pore_number=i, pore_size=particle.get('Number of pixels')
            )
            pdf.savefig(bbox_inches='tight')
        
        combined_fit_params = extract_param_values_from_fit_model(
            combined_fit_model, 
            ['step_sigma', 'overshoot_sigma', 'undershoot_sigma', 'step_amplitude', 'overshoot_amplitude', 'undershoot_amplitude']
        )
        particle_data.append({
            'Slice': slice_index,
            'Component': particle.get('Component'),
            'Pore CNR' : pore_CNR,
            'Number of pixels': particle.get('Number of pixels'),
            'initial-fit sigma': extract_param_values_from_fit_model(init_fit_model, ['step_sigma']).get('step_sigma'),
            'initial-fit amplitude': extract_param_values_from_fit_model(init_fit_model, ['step_amplitude']).get('step_amplitude'),
            **combined_fit_params
        })
    return particle_data

def generate_boxplot(box_plot, slice_indexes, number_of_slices, pdf):
    """Generate and save a boxplot of particle sizes."""
    import matplotlib.pyplot as plt
    plt.boxplot(box_plot, showfliers=False, positions=slice_indexes, vert=True, widths=200 / number_of_slices, patch_artist=True)
    plt.yscale('log')
    plt.title('Segmented Particle Size Boxplot Over Slices')
    pdf.savefig(bbox_inches='tight')

def filter_dataframe(df, max_sigma):
    """Filter the DataFrame to remove outliers based on sigma."""
    columns_to_check = ['step_sigma', 'overshoot_sigma', 'undershoot_sigma', 'initial-fit sigma']
    return df[(df[columns_to_check].abs() <= max_sigma).all(axis=1)]

def generate_final_report(df, slice_CNR, save_path, file_name):
    """
    Generate the final report summarizing weighted sigmas and plots and save it to a PDF.
    
    Args:
        df (pd.DataFrame): DataFrame containing analysis results with necessary columns for weighted sigma calculation.
        save_path (str): Path to save the final report.
        file_name (str): Name of the sample for use in the report file name and annotations.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from tabulate import tabulate
    import scipy.stats
    import numpy as np

    # Define the PDF file path for the report
    pdf_file_path = os.path.join(save_path, f'IQ_Overview_report_Sample_{file_name}.pdf')

    # Ensure required columns exist in the DataFrame
    required_columns = ['Slice', 'step_sigma', 'step_amplitude', 
                        'overshoot_sigma', 'overshoot_amplitude', 
                        'undershoot_sigma', 'undershoot_amplitude', 
                        'initial-fit sigma']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in DataFrame: {col}")

    # Calculate weighted sigmas for each slice
    per_slice_results = (
        df.groupby('Slice')
        .apply(lambda group: pd.Series({
            'Weighted Sum Sigma': (
                (abs(group['step_sigma']) * abs(group['step_amplitude'])).sum() +
                (abs(group['overshoot_sigma']) * abs(group['overshoot_amplitude'])).sum() +
                (abs(group['undershoot_sigma']) * abs(group['undershoot_amplitude'])).sum()
            ) / (
                abs(group['step_amplitude']).sum() +
                abs(group['overshoot_amplitude']).sum() +
                abs(group['undershoot_amplitude']).sum()
            ),
            'Weighted initial-fit Sigma': (
                (abs(group['initial-fit sigma']) * abs(group['step_amplitude'])).sum()
            ) / abs(group['step_amplitude']).sum(),
            'Weighted comb-fit Sigma': (
                (abs(group['step_sigma']) * abs(group['step_amplitude'])).sum()
            ) / abs(group['step_amplitude']).sum()
        }))
    ).reset_index()

    # Calculate overall weighted averages
    overall_results = pd.Series({
        'Weighted Sum Sigma': (
            (abs(df['step_sigma']) * abs(df['step_amplitude'])).sum() +
            (abs(df['overshoot_sigma']) * abs(df['overshoot_amplitude'])).sum() +
            (abs(df['undershoot_sigma']) * abs(df['undershoot_amplitude'])).sum()
        ) / (
            abs(df['step_amplitude']).sum() +
            abs(df['overshoot_amplitude']).sum() +
            abs(df['undershoot_amplitude']).sum()
        ),
        'Weighted initial-fit Sigma': (
            (abs(df['initial-fit sigma']) * abs(df['step_amplitude'])).sum()
        ) / abs(df['step_amplitude']).sum(),
        'Weighted comb-fit Sigma': (
            (abs(df['step_sigma']) * abs(df['step_amplitude'])).sum()
        ) / abs(df['step_amplitude']).sum()
    })

    # Save to PDF
    with PdfPages(pdf_file_path) as pdf:
        # add slice CNR to per_slice_results dataframe
        # Map slice CNR to the actual slices present in per_slice_results
        slice_cnr_map = {i: f"{v:.2f}" for i, v in zip(sorted(set(per_slice_results['Slice'])), slice_CNR)}
        per_slice_results['Slice CNR'] = per_slice_results['Slice'].map(slice_cnr_map)
        print(per_slice_results)
        # Add textual summary
        text_summary = (
            #f'Slice CNR:\n{"\n".join(f"{f:.2f}" for f in slice_CNR)}'
            f'Overall Results:\n{overall_results.round(2)}\n\n'
            f'Per Slice Results:\n{tabulate(per_slice_results.round(2), headers="keys", tablefmt="pretty")}'
        )
        
        # Dynamically calculate the figure size based on rows in the per_slice_results
        fig_height = len(per_slice_results) * 0.2  # Adjust multiplier for more/less space per line
        fig_width = 8.5  # Standard width of a page
        
        # Create figure and axis for text
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")  # Turn off axis
        ax.text(0, 1, text_summary, fontsize=10, ha='left', va='top', wrap=True)
        #plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')  # Save text as a page
        fig.clear()
        plt.close(fig)

        # Plot weighted sigmas per slice
        slices = per_slice_results['Slice']
        weighted_sum_sigmas = per_slice_results['Weighted Sum Sigma']
        weighted_initial_fit_sigmas = per_slice_results['Weighted initial-fit Sigma']
        weighted_comb_fit_sigmas = per_slice_results['Weighted comb-fit Sigma']

        # Overall weighted averages
        overall_weighted_sum_sigma = overall_results['Weighted Sum Sigma']
        overall_weighted_initial_fit_sigma = overall_results['Weighted initial-fit Sigma']
        overall_weighted_comb_fit_sigma = overall_results['Weighted comb-fit Sigma']

        # Create sigma plot
        plt.figure(figsize=(12, 8))
        plt.plot(slices, weighted_sum_sigmas, label='Weighted Sum Sigma', marker='o', color='blue')
        plt.plot(slices, weighted_initial_fit_sigmas, label='Weighted Initial-Fit Sigma', marker='s', color='green')
        plt.plot(slices, weighted_comb_fit_sigmas, label='Weighted Comb-Fit Sigma', marker='^', color='orange')

        # Add overall averages as horizontal lines
        plt.axhline(y=overall_weighted_sum_sigma, color='blue', linestyle='--', label='Overall Weighted Sum Sigma')
        plt.axhline(y=overall_weighted_initial_fit_sigma, color='green', linestyle='--', label='Overall Weighted Initial-Fit Sigma')
        plt.axhline(y=overall_weighted_comb_fit_sigma, color='orange', linestyle='--', label='Overall Weighted Comb-Fit Sigma')

        # Customize plot
        plt.title('Weighted Sigmas Per Slice with Overall Averages', fontsize=16)
        plt.xlabel('Slice', fontsize=14)
        plt.ylabel('Sigma Values', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.6)
        plt.tight_layout()
        pdf.savefig()  # Save plot to PDF
        plt.close()
        
        # plot the slicewise CNR
        plt.figure(figsize=(12, 8))
        plt.plot(slice_CNR, label='Slice CNR', marker='o', color='blue')
        # Add overall averages as horizontal lines
        plt.axhline(y=np.average(np.asarray(slice_CNR)), color='blue', linestyle='--', label='Average Slice CNR')
        # Customize plot
        plt.title('CNR over Slice', fontsize=16)
        #plt.xlabel('Slice', fontsize=14)
        plt.ylabel('CNR [1]', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.6)
        plt.tight_layout()
        pdf.savefig()  # Save plot to PDF
        plt.close()
        

        # Plot Pearson correlation scatter plot
        try:
            corr, p_value = scipy.stats.pearsonr(df['Slice'], df['step_sigma'])
            plt.figure(figsize=(8, 6))
            plt.scatter(df['Slice'], df['step_sigma'], label='Data Points', alpha=0.7, edgecolor='k')

            # Fit and plot regression line
            coeffs = np.polyfit(df['Slice'], df['step_sigma'], 1)
            line = np.polyval(coeffs, df['Slice'])
            plt.plot(df['Slice'], line, color='red', label='Regression Line')

            # Annotate correlation coefficient
            plt.text(0.05, 0.95, f'Pearson r: {corr:.2f}\nP-value: {p_value:.2e}',
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

            # Customize scatter plot
            plt.title('Scatter Plot with Pearson Correlation', fontsize=16)
            plt.xlabel('Slice', fontsize=14)
            plt.ylabel('Comb-Fit Sigma [px]', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            pdf.savefig()  # Save plot to PDF
            plt.close()
        except:
            print("not enough points for a pearson-correlation. Skipped")
    print(f"Final report saved to {pdf_file_path}")

#print('Welcome. Please call the following function via console:\nprintIQreport(file_path=None, mask_path=None, save_path=None, number_of_slices=None, margin=0, min_particle_size=10, max_particle_size=1000, Pores_to_analyze=3, cnr_mask_erosion_value=None, bright_pores=False)')

if __name__ == "__main__":
    import sys
    import argparse
    
    #print(sys.argv)
    #raise NotImplementedError('Module not yet usable from command line, please import into a python interpreter, and use function printIQReport')
    # ~ import pdb
    # ~ pdb.set_trace()
        

    import re
    import os
    save_path = '/mnt/XNAS/data/46_AIQuAM3D/99_code/Collection_Testrun_Anatomix_112024/tmp/'
    file_path = '/mnt/XNAS/data/46_AIQuAM3D/01_beamtimes/2024_12_Anatomix/'
    save_path = os.getcwd()+'\\'#'Z:/46_AIQuAM3D/99_code/Collection_Testrun_Anatomix_112024/tmp/'
    file_path = save_path + 'TestROI_uint8_1204x976x50.raw'#'Z:/46_AIQuAM3D/01_beamtimes/2024_12_Anatomix/'
    save_path += 'TestROI_IQreport/'#'Z:/46_AIQuAM3D/99_code/Collection_Testrun_Anatomix_112024/tmp/'
    #single_file = save_path + 'TestROI_uint8_1204x976x50.raw'

    printIQreport(
        file_path=file_path, 
        mask_path=None, 
        save_path=save_path, 
        number_of_slices=5, 
        margin=25, 
        min_particle_size=None, 
        max_particle_size=None, 
        Pores_to_analyze=3, 
        cnr_mask_erosion_value=None, 
        bright_pores=False, 
        other_label_exclusion_margin=4,
        local_tomo=True
    )

    #for single_file in os.listdir(file_path):
    #	if single_file.endswith('.raw'):
    #		sample_name = re.split(r'(_uint8_|_float32_|_uint16_)', os.path.basename(single_file))[0]
    #		printIQreport(file_path+single_file, save_path=save_path, margin=25, number_of_slices=5, local_tomo=True)
