#this file is a collection of the image processing functions of Q3 i've initialy solved using PyIPSDK.
#Due to the unavailability at some locations here are the open-source solutions i was able to come up with.
#I'm assuming that the volumes handled are np.arrays


#List of functions that should be found here in the end:
'''
border_point_calculation_open_source - takes a volume and a label volume and returns the intensity values of 
readRawVolume - use fileIO
saveRawVolume - use fileIO
calculate_otsu_threshold - using skimage.filters
threshold_volume - using numpy
fill_hole3d - using scipy binary_fill_holes
get3dROI - using arrays
connected_component_3d - using scipy.ndimage.label
labelAnalysis(NbPixels, boundingbox) - i've implemented a filtering by sphericity and NbPixels
calculate CNR - using numpy
3d distance map -using scipy
'''

import numpy as np
from skimage import filters
from scipy.ndimage import binary_fill_holes, label, find_objects, binary_erosion, distance_transform_edt, binary_dilation
import matplotlib.pyplot as plt

def border_point_calculation_open_source(volume, label_volume, label_id, MAX_DIST=25, other_label_exclusion_margin=5, dark_particles=True):
    '''
    output = border_point_calculation(mask, volume)
    _______________________________________________________________________________________________
    Input:
    - mask      ; type PyIPSDK mask  	; mask of the element to calculate the border points of
    - volume    ; type PyIPSDK image 	; greyscale volume of the same region and size
    - MAX_DIST	; tpye int				; maximum distance to calculate the border points of
    _______________________________________________________________________________________________
    Output:
    two np.ndarrays of the distance to the surface and the average greyvalue at that position
    _______________________________________________________________________________________________
    '''
    import math
    mask = threshold(label_volume, label_id, label_id).astype(np.bool_)
    other_label_exclusion = create_other_label_exclusion_mask(label_volume, label_id, margin=other_label_exclusion_margin)
    dark_to_bright = 1 if dark_particles is True else -1
    
    dm3d_i 	= calculate_3d_distance_map(mask, 0)
    dm3d 	= calculate_3d_distance_map(mask, 1)
    
    values = []
    for i in reversed(np.unique(dm3d)):
        if i != 0 and i <= MAX_DIST:
            dist_mask = threshold(dm3d, float(i), float(i))
            masked_values = volume[(dist_mask == 1) & (other_label_exclusion)]
            if np.any(masked_values):
                value = np.nanmedian(masked_values)#np.nanmean(masked_values)#nanmean nanmedian average?
                values.append([float(i)*-1*dark_to_bright+1,float(value)])
    for i in np.unique(dm3d_i):
        if (i != 0 and i <=MAX_DIST):
            dist_mask =  threshold(dm3d_i, float(i), float(i))
            masked_values = volume[(dist_mask == 1) & (other_label_exclusion)]
            if np.any(masked_values):
                value = np.nanmedian(masked_values)# np.nanmean(masked_values)
                values.append([float(i)*dark_to_bright,float(value)])
    if not values:
        #raise ValueError('Border point calculation at a point yielded no values: returning [0,0]')
        return np.asarray([0,0]), np.asarray([0,0])
    else:
        x, y = zip(*np.nan_to_num(values))
        return np.asarray(x), np.asarray(y)


def create_other_label_exclusion_mask(label_volume, label_id, margin=5):
	'''
	takes in a label_volume and the label_id of the label of interest. returns a mask that puts a margin around every other label in the volume
	'''
	# Create a mask for all labels other than the label_id and the background (0)
	other_label_volume = (label_volume != label_id) & (label_volume != 0)
	# Check if there are any other labels besides label_id and background
	if np.sum(other_label_volume) == 0:
        # If no other labels are present, return a mask filled with True
		return np.ones(label_volume.shape, dtype=bool)
    
    # Calculate the distance map from other labels
	distance_map = calculate_3d_distance_map(threshold(other_label_volume, 0, 0))
    
    # Return the mask where the distance from other labels is greater than the specified margin
	return threshold(distance_map, 0, margin, inverse=True).astype(np.bool_)

def get_equally_spaced_integers(start, end, num_points, exclude_outside=False):
    if exclude_outside:
        return np.linspace(start, end, num_points+2, dtype=int).tolist()[1:-1]
    else:
	    return np.linspace(start, end, num_points, dtype=int).tolist()

def get_slice(volume, index=None):
	'''
	returns a single slice of the volume, default at center in z
	'''
	if index is None:
		index = volume.shape[0]//2
	return volume[index, :, :]

def get_orth_slice(volume, index=None, axis='z'):
    '''
    returns a single-ortho slice in the given direction.
    '''
    axis = 2 if axis=='x' else 1 if axis=='y' else 2
    if index is None:
        index = volume.shape[axis]//2
    return get_ROI(volume, ())
 

def show_slice(volume, offset=0):
	'''
	Shows center slice (+ offset) in z of the given volume as a matplotlib-plot 
	'''
	import matplotlib.pyplot as plt 
	plt.imshow(volume[volume.shape[0]//2+offset, : ,: ])
	plt.show()

def show(image):
	plt.imshow(image)
	plt.show()

def calculate_3d_distance_map(mask, label_to_calculate_dm3d_on=1, dmap_type=np.uint8):
	'''
	Calculates the eucledian distance from the surface of the masked volume.

	_______________________________________________________________________________________________
	Input:
	- mask	; numpy array	; mask to calculate the distance_map of
	- label_to_calculate_dm3d_on ; int ;
	- dmap_type ; np fileformat ; datatype to return distance map in
	_______________________________________________________________________________________________
	Output:
	np.array with the eucledian distance map of the volume
	_______________________________________________________________________________________________
	'''
	mask = np.asarray(mask).astype(np.bool_)
	
	distance_map = distance_transform_edt(mask == label_to_calculate_dm3d_on)
	
	if dmap_type is not np.float32:
		return np.round(distance_map).astype(dmap_type)
	else:
		return distance_map.astype(dmap_type)
		
def calc_3d_CNR_open_source(vol, mask_1, mask_2=None, erosion_value=None):
	'''
	output = calc_3d_CNR_PyIPSDK(vol, mask)

	_______________________________________________________________________________________________
	Input:
	- vol	; numpy array	; greyscale volume to calculate the CNR for
	- mask	; numpy array	; mask seperating the two phases to calculate the CNR for
	_______________________________________________________________________________________________
	Output:
	CNR as a floating point number
	_______________________________________________________________________________________________

	'''
	mask = mask_1.astype(np.bool_)
	if erosion_value is not None:
		mask = erode(mask, erosion_value)
    
	
	#stats of masked area:
	vol_masked = vol[mask]
	mean = np.nanmean(vol_masked)
	std_dev = np.std(vol_masked)
	
	#stats of the inverted masked area:
	if mask_2 is None:
		#print('No mask provided for the second Phase for CNR-calculation. Using inverse of mask provided.')
		mask = ~mask_1-254 #-254 since the iversion is 255 for 1 and 254 for 0 when inverting an uint8 array
	else:
		mask = mask_2
	if erosion_value is not None:
		mask = erode(mask, erosion_value)
	vol_masked_i = vol[mask.astype(np.bool_)]
	mean_i = np.nanmean(vol_masked_i)
	std_dev_i = np.std(vol_masked_i)
	return abs(mean - mean_i)/(np.nan if (std_dev if std_dev >= std_dev_i else std_dev_i) == 0.0 else 
                            (std_dev if std_dev >= std_dev_i else std_dev_i))

def erode(mask, erosion_value=1, structure=None):
    '''
    Erode a binary mask by a given value. Optionally a structuring element can be specified . If none is given an element with a square connectivitiy equal to one is created.
    output = erode(mask, erosion_value, structure)
    _______________________________________________________________________________________________
	Input:
	- mask	; numpy array	; mask seperating the two phases to calculate the CNR for
    - erosion_value; int    ; value to erode the mask by
    - structure; array_like ; struction element used for the erosion (see scipy.ndimage.binary_erosion)
	_______________________________________________________________________________________________
	Output:
	eroded binary mask of input.
	_______________________________________________________________________________________________
    '''
    return binary_erosion(mask.astype(np.bool_), iterations=int(erosion_value), structure=structure)

def dilate(mask, dilation_value=1, structure=None):
    '''
    Erode a binary mask by a given value. Optionally a structuring element can be specified . If none is given an element with a square connectivitiy equal to one is created.
    output = erode(mask, erosion_value, structure)
    _______________________________________________________________________________________________
	Input:
	- mask	; numpy array	; mask seperating the two phases to calculate the CNR for
    - dilation_value; int   ; value to dilate the mask by
    - structure; array_like ; struction element used for the erosion (see scipy.ndimage.binary_dilatin)
	_______________________________________________________________________________________________
	Output:
	dilated binary mask of input.
	_______________________________________________________________________________________________
    '''
    return binary_dilation(mask.astype(np.bool_), iterations=int(dilation_value), structure=structure)

def connected_component_analysis(labeled_volume, number_of_components=None):
    '''
    Analyze connected components in a labeled volume, supporting both 2D and 3D inputs.
    
    _______________________________________________________________________________________________
    Input:
    - labeled_volume        : numpy array of label image (2D or 3D)
    - number_of_components  : int (optional) - Number of labels in labeled_volume, calculated if not provided.
    _______________________________________________________________________________________________
    Output:
    - List of dictionaries containing:
        - 'Component'        : label of the component  : int
        - 'Number of pixels' : number of pixels in the component  : int
        - 'Bounding Box'     : coordinates of the bounding box in the form:
                               - For 2D: (xmin, xmax, ymin, ymax)
                               - For 3D: (xmin, xmax, ymin, ymax, zmin, zmax)
        - 'Sphericity'       : sphericity calculated as: (6 * Volume)^(1/3) / Surface_Area  : float
    _______________________________________________________________________________________________
    '''
    
    # Check the dimensionality of the input (2D or 3D)
    is_3d = len(labeled_volume.shape) == 3
    
    # Get the number_of_components from labeled_volume if not provided
    if number_of_components is None:
        number_of_components = labeled_volume.max()

    results = []
    for component in range(1, number_of_components + 1):
        component_mask = (labeled_volume == component)

        # Get number of pixels in the component
        NbPixels = np.sum(component_mask)

        # Calculate the bounding box using find_objects
        slices = find_objects(labeled_volume == component)
        if slices:
            bbox_slices = slices[0]
            if is_3d:
                # 3D case
                bbox_x_min, bbox_x_max = bbox_slices[0].start, bbox_slices[0].stop
                bbox_y_min, bbox_y_max = bbox_slices[1].start, bbox_slices[1].stop
                bbox_z_min, bbox_z_max = bbox_slices[2].start, bbox_slices[2].stop
                bounding_box = (bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max)
            else:
                # 2D case
                bbox_x_min, bbox_x_max = bbox_slices[0].start, bbox_slices[0].stop
                bbox_y_min, bbox_y_max = bbox_slices[1].start, bbox_slices[1].stop
                bounding_box = (bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max)
        else:
            bounding_box = (0, 0, 0, 0) if not is_3d else (0, 0, 0, 0, 0, 0)

        # Calculate sphericity
        # Assumption: surface ~ number of pixels when eroded by 1
        eroded_mask = binary_erosion(component_mask)
        surface_area = np.sum(eroded_mask)
        sphericity = (6 * NbPixels) ** (1/3) / surface_area if surface_area > 0 else 0

        # Append results for each component
        results.append({
            'Component': component,
            'Number of pixels': NbPixels,
            'Bounding Box': bounding_box,
            'Sphericity': sphericity
        })
    
    return results


def filter_and_sort_components(components, volume_shape, top_n=None, min_size=None, max_size=None, sphericity_threshold=None, min_distance_from_volume_edges=None, min_bbox_size=None):
    """
    Filters and sorts the connected components based on size, sphericity, and bounding box constraints.
	_______________________________________________________________________________________________
    filtered_cc_analysis  = filter_and_sort_components(cc_analysis)
    _______________________________________________________________________________________________
    Input:
    - components               : type list of dict ; list of dictionaries containing connected component analysis results
    - top_n                    : type int          ; number of top components to return based on size
    - sphericity_threshold      : type float        ; minimum sphericity value required to include a component
    - volume_shape              : tuple             ; shape of the volume (2D or 3D)
    - min_distance_from_volume_edges : int          ; minimum distance of the bounding box from the volume edges
    - min_bbox_size             : int               ; minimum size of the bounding box in any dimension
    ______________________________________________________________________________________________
    Output:
    - list of dictionaries containing filtered and sorted by number of pixels in descending order of the connected component analysis results
    _______________________________________________________________________________________________
    """
    
    if volume_shape is None:
        raise ValueError("volume_shape must be provided to filter based on proximity to borders.")
    
    # Check if volume is 2D or 3D
    is_3d = len(volume_shape) == 3
    
    def is_valid_component(comp):
        """Helper function to filter based on proximity to borders and bounding box size."""
        
        if is_3d:
            # 3D bounding box and checks
            bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max = comp['Bounding Box']

            # Check proximity to volume edges
            if min_distance_from_volume_edges is not None:
                if (bbox_x_min < min_distance_from_volume_edges or bbox_x_max > volume_shape[0] - min_distance_from_volume_edges or
                    bbox_y_min < min_distance_from_volume_edges or bbox_y_max > volume_shape[1] - min_distance_from_volume_edges or
                    bbox_z_min < min_distance_from_volume_edges or bbox_z_max > volume_shape[2] - min_distance_from_volume_edges):
                    return False

            # Check minimum bounding box size
            if min_bbox_size is not None:
                if (bbox_x_max - bbox_x_min < min_bbox_size or
                    bbox_y_max - bbox_y_min < min_bbox_size or
                    bbox_z_max - bbox_z_min < min_bbox_size):
                    return False
        
        else:
            # 2D bounding box and checks
            bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = comp['Bounding Box']

            # Check proximity to volume edges
            if min_distance_from_volume_edges is not None:
                if (bbox_x_min < min_distance_from_volume_edges or bbox_x_max > volume_shape[0] - min_distance_from_volume_edges or
                    bbox_y_min < min_distance_from_volume_edges or bbox_y_max > volume_shape[1] - min_distance_from_volume_edges):
                    return False

            # Check minimum bounding box size
            if min_bbox_size is not None:
                if (bbox_x_max - bbox_x_min < min_bbox_size or
                    bbox_y_max - bbox_y_min < min_bbox_size):
                    return False

        # Check sphericity threshold
        if sphericity_threshold is not None:
            if comp['Sphericity'] <= sphericity_threshold:
                return False

        # Check size threshold
        if min_size is not None or max_size is not None:
            component_size = comp['Number of pixels'] if not is_3d else comp['Number of voxels']    
            if min_size is not None:
                if component_size < min_size:
                    return False
            if max_size is not None:        
                if component_size > max_size:
                    return False

        return True

    # Filter components based on sphericity, proximity to borders, and bounding box size
    filtered_components = [comp for comp in components if is_valid_component(comp)]

    # Sort components by the number of pixels in descending order
    sorted_components = sorted(filtered_components, key=lambda x: x['Number of pixels'], reverse=True)

    # Return the top N components
    if top_n is not None:
        return sorted_components[:top_n]
    else:
        return sorted_components

def connected_component_3d(mask):
	'''
	Creates a labelimage of connected components in a 3d binary volume.
	_______________________________________________________________________________________________
    cc3d  = connected_component_3d(mask)
    _______________________________________________________________________________________________
    Input:
    - mask		; type np.array	; binary array to create the connected component on
    ______________________________________________________________________________________________
    Output:
    - connected_component label np.array
    _______________________________________________________________________________________________
	'''
	#make sure the mask is binary/boolean array
	mask = (mask > 0).astype(np.uint8)
	
	#connected component using scipy.ndimage.label
	labeled_volume, number_of_components = label(mask)
	
	return labeled_volume, number_of_components

def get_ROI(volume, bb_coords, margin=0):
    '''
    Extract a sub-volume from a larger numpy array (2D or 3D).
        
    _______________________________________________________________________________________________
    Input:
    - volume    ; type np.array                ; input array (2D or 3D)
    - bb_coords ; type list or tuple of int     ; coordinates of the bounding box
                  ; format (xmin, xmax, ymin, ymax) for 2D
                  ; format (xmin, xmax, ymin, ymax, zmin, zmax) for 3D
    - margin     ; type int                      ; (optional) margin taken around the ROI if possible as padding.
    ______________________________________________________________________________________________
    Output:
    - subVolume  ; np.array                     ; extracted sub-volume with margin
    _______________________________________________________________________________________________
    '''
    
    # Check the dimensionality of the volume
    is_3d = len(volume.shape) == 3
    
    if is_3d:
        xmin, xmax, ymin, ymax, zmin, zmax = bb_coords
        # Validate that the bounding box is within the original volume dimensions
        if (xmin < 0 or ymin < 0 or zmin < 0 or
            xmax > volume.shape[0] or
            ymax > volume.shape[1] or
            zmax > volume.shape[2]):
            raise ValueError("SubVolume dimensions are out of bounds.")
        
        # Apply the margin, ensuring we don't go out of bounds
        xmin_with_margin = max(xmin - margin, 0)
        xmax_with_margin = min(xmax + margin, volume.shape[0])

        ymin_with_margin = max(ymin - margin, 0)
        ymax_with_margin = min(ymax + margin, volume.shape[1])

        zmin_with_margin = max(zmin - margin, 0)
        zmax_with_margin = min(zmax + margin, volume.shape[2])

        # Extract the sub-volume with the margin
        subVolume = volume[xmin_with_margin:xmax_with_margin, 
                           ymin_with_margin:ymax_with_margin, 
                           zmin_with_margin:zmax_with_margin]

    else:
        xmin, xmax, ymin, ymax = bb_coords
        # Validate that the bounding box is within the original volume dimensions
        if (xmin < 0 or ymin < 0 or
            xmax > volume.shape[0] or
            ymax > volume.shape[1]):
            raise ValueError("SubVolume dimensions are out of bounds.")

        # Apply the margin, ensuring we don't go out of bounds
        xmin_with_margin = max(xmin - margin, 0)
        xmax_with_margin = min(xmax + margin, volume.shape[0])

        ymin_with_margin = max(ymin - margin, 0)
        ymax_with_margin = min(ymax + margin, volume.shape[1])

        # Extract the sub-volume with the margin
        subVolume = volume[xmin_with_margin:xmax_with_margin, 
                           ymin_with_margin:ymax_with_margin]

    return subVolume


def fill_holes(mask):
	'''
	Fill holes in a binary np array. (using scipy.ndimage.binary_fill_holes)
	
	_______________________________________________________________________________________________
    mask_holes_filled = fill_holes(mask)
    _______________________________________________________________________________________________
    Input:
    - mask		; type np.array	; binary array containing a mask to fill
    ______________________________________________________________________________________________
    Output:
    - filled_mask np.array
    _______________________________________________________________________________________________
	'''
	filled_volume = binary_fill_holes(mask)
	return filled_volume.astype(np.uint8)

def calculate_otsu_threshold(volume, mask=None):
	'''
	Caclulate the otsu-threshold value of a (otionally masked) volume. (using skimage.filters.threshold_otsu)
	
	_______________________________________________________________________________________________
    otsu_value = calculate_otsu_threshold(volume, mask=None)
    _______________________________________________________________________________________________
    Input:
    - volume	; type np.array	; array containing the greyscale volume to calculate the otsu-threshold value of
    - mask		; type np.array	; (optional) array containing a mask to restrict the values to calculate the otsu value on
    ______________________________________________________________________________________________
    Output:
    - otsu_value 
    _______________________________________________________________________________________________
	'''
	volume = np.asarray(volume)
	if mask is not None:
		mask = np.asarray(mask).astype(bool)
		masked_volume = volume[mask]
	else:
		masked_volume = volume
	otsu_value = filters.threshold_otsu(masked_volume)
	
	return otsu_value

def create_circular_mask(image: np.ndarray, reduction: int = 5) -> np.ndarray:
	'''
	create a circular mask at the center of a 2d image with a diameter of the shorter axis minus the reduction value.
		
	_______________________________________________________________________________________________
	result = create_circular_mask(image, reduction=5) 
    _______________________________________________________________________________________________
    Input:
    - image			; type np.array	; array containing the greyscale volume to create a mask for
    - reduction		; int			; value of the radius reduction
    ______________________________________________________________________________________________
    Output:
    - np.array of the thresholded array
    _______________________________________________________________________________________________
	'''
	height, width = image.shape[:2]
	min_axis = min(height, width) - reduction
	radius = min_axis // 2
    
	y, x = np.ogrid[:height, :width]
	center_y, center_x = height // 2, width // 2
	mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    
	return mask


def threshold(volume, threshold_value_low,  threshold_value_high=None, inverse=False):
	'''    
	Threshold a volume with a given value. Get background image if keep_lower is set to true. (using numpy)
	
	_______________________________________________________________________________________________
	result = threshold(volume, threshold_value, keep_lower=True)
    _______________________________________________________________________________________________
    Input:
    - volume			; type np.array	; array containing the greyscale volume to threshold
    - threshold_value	; int or float	; value to threshold at
    - inverse			; boolean		; (optional) if set to true the background image is returned
    ______________________________________________________________________________________________
    Output:
    - np.array of the thresholded array
    _______________________________________________________________________________________________
	'''
	volume = np.asarray(volume)
    
	if threshold_value_high is None:
        # Single threshold case (same behavior as before)
		if inverse:
			thresholded_volume = (volume <= threshold_value_low).astype(np.uint8)
		else:
			thresholded_volume = (volume > threshold_value_low).astype(np.uint8)
	else:
		# Double threshold case (threshold between low and high)
		if inverse:
			# Inverse: Keep values outside the threshold range
			thresholded_volume = ((volume < threshold_value_low) | (volume > threshold_value_high)).astype(np.uint8)
		else:
			# Normal: Keep values inside the threshold range
			thresholded_volume = ((volume >= threshold_value_low) & (volume <= threshold_value_high)).astype(np.uint8)

	return thresholded_volume
