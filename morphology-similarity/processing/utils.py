"""
Contains utility functions for the preprocessing of images for later
analysis.
"""
import processing
import numpy as np
from skimage import measure


def extract_components(image, 
                       binarize=True, 
                       background=-1,
                       allow_blank_images=False,
                       return_images=False):
    """
    Extract morpholgoical components from an image.

    Arguments:
        image (ndarray): 
            A grayscale image 
        binarize (boolean): 
            Flag to binarize the image before extraction.Defaults to
            True.
        background (int): 
            Pixels of this value will be considered background and will
            not be labeled. By default every pixel is considered for
            labeling. Set to -1 for no background. (default=-1)
        allow_blank_images (boolean):
            Whether a blank image should be considered as a single
            component. (default=False) 
        return_images (boolean): 
            Wheter to return the labeled image & binary
            image.(default=False)

    Returns:
        components (list): 
            A list of component images with the same shape as the input
            image.
        image (ndarray): 
            Original image (and binarized if binarize=True). Only
            returned when return_images is set to True.
        labeled_sample (ndarray): 
            A labelled image where all connected ,regions are assigned
            the same integer value. Only returned when return_images is
            set to True.
    """
    components = []
    if binarize:
        image = (image > 0.5).astype(int)

    labeled_sample = measure.label(image, background=background)

    for label in np.unique(labeled_sample):
        # extract companent into a separate image
        component = (labeled_sample == label).astype(np.float64)
        
        if not allow_blank_images:
            if not (component == 0).all():
                continue 
                
        components.append(component)

    # remove the first background component if background pixels needs
    # to be neglected. 
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#label
    if background >= 0:
        components = components[1:]
    
    if return_images:
        return components, image, labeled_sample
    else:
        return components


def crop_image(image, tolerance=0):
    """
    Crop an image by reducing the dimensions to a minimal size
    while ensuring all non-zero pixel are present.

    Arguments:
        image: An image, represented as a 2D NumPy array
        tolerance: A float for masking tolerance.  Defaults to 0.

    Returns:
        An image of variable shape.
    """
    mask = image > tolerance
    return image[np.ix_(mask.any(1), mask.any(0))]


def apply_signatures(image, sig_funcs, allow_empty_sig=False):
    """
    Applies the provided signature functions to the given image.

    Arguments:
        image: An image, represented as a 2D Numpy array.
        sig_funcs: List of signature extraction functions.
        allow_empty_sig: If signature values of zero should be allowed
                         (default=False)

    Returns:
        A list of signatures for the given image.

    Raises:
        AssertionError: All signatures returned by the extractors need
        to be non-empty. Only when allow_empty_sig is False.
    """
    if isinstance(sig_funcs, str):
        # For convenience, we can pass in a single signature function.
        # This converts it into a list, with it being the only element.
        sig_funcs = [sig_funcs]
    sigs = []
    for sig_func in sig_funcs:
        sig = eval("processing."+sig_func+"(image)")
        if not allow_empty_sig:
            assert len(sig) > 0, 'Signatures need to be non-empty.'
        sigs.append(sig)
    sigs = np.array(sigs).T
    return sigs


def apply_to_components(image, func, crop=True):
    """
    Apply the given function to the components of each image.

    Arguments:
        image: An image, represented as a 2D NumPy array
        func: The function to apply to the components
        crop: Flag to crop the component to minimal dimensions (defaults to True)

    Returns:
        A list of measurements corresponding to the result of the function call on each component.
    """
    measurements = []
    components = extract_components(image)
    for component in components:
        if crop:
            component = crop_image(component)
        measurement = func(component)
        measurements.append(measurement)
    return np.array(measurements)
