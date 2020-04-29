"""
Contains utility functions for the preprocessing of images for later analysis.
"""

import numpy as np
from skimage import measure


def extract_components(image, binarize=True, background=-1):
    """
    Extract morpholgoical components from an image.

    Arguments:
        image: An image, represented as a 2D NumPy array
        binarize: Flag to binarize the image before extraction.  Defaults to True.

    Returns:
        A list of component images with the same shape as the input image
    """
    components = []
    if binarize:
        image = (image > 0.5).astype(int)
    labeled_sample = measure.label(image, background=background)
    for component in np.unique(labeled_sample):
        if not (component == 0).all():
            # if the entire image isn't all blank, collect it
            components.append(component)
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


def apply_signatures(image, sig_funcs):
    """
    Applies the provided signature functions to the given image.

    Arguments:
        image: An image, represented as a 2D Numpy array.
        sig_funcs: List of signature extraction functions.

    Returns:
        A list of signatures for the given image.

    Raises:
        AssertionError: All signatures returned by the extractors need to be non-empty.
    """
    if isinstance(sig_funcs, list):
        # For convenience, we can pass in a single signature function.
        # This converts it into a list, with it being the only element.
        sig_funcs = [sig_funcs]
    sigs = []
    for sig_func in sig_funcs:
        sig = sig_func(image)
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
