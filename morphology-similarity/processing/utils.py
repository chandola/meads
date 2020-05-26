"""
Contains utility functions for the preprocessing of images for later
analysis.
"""
import processing
import numpy as np
from skimage import measure


def binarize(image, thresh=0.5):
    return (image > thresh).astype(int)


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
            if (component == 0).all():
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


def extract_minkowski_scalars(image, binarize=True, background=-1):
    """
    Extract 2D Minkowski scalars for each component of an image.

    Arguments:
        image: An image, represented as a 2D NumPy array
        binarize: Flag to binarize the image before extraction.  Defaults to True.

    Returns:
        A list of Minkowski signatures for each component in the image
    """
    component_signatures = []
    if binarize:
        image = (image > 0.5).astype(int)
    labeled_sample = measure.label(image, background=background)
    region_props = measure.regionprops(labeled_sample)
    for props in region_props:
        component = np.array([
            props.area,
            props.perimeter,
            props.euler_number
        ])
        component_signatures.append(component)
    return np.array(component_signatures)


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


def get_inscribed_rect_area(component):
    """
    Calculate the area of the largest rectangle inscribed
    in the given component.
    (adapted from https://stackoverflow.com/a/30418912)

    Arguments:
        component: An image, represented as a 2D NumPy array

    Returns:
        The tuple whose first component is the area, as a float,
        and whose second component is a tuple representing the
        bounding coordinates of the rectangle.
    """
    nrows = component.shape[0]
    ncols = component.shape[1]
    skip = 0
    area_max = (0, [])

    a = component
    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)
    for r in range(nrows):
        for c in range(ncols):
            if a[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r-1][c]+1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c-1]+1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh+1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, c-minw+1, r, c)])
    return area_max


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


def apply_to_image(image, func, crop=True):
    """
    Apply the given function to an image.

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


def apply_to_components(components, func, crop=True):
    """
    Apply the given function to the components of an image.

    Arguments:
        image: An image, represented as a 2D NumPy array
        func: The function to apply to the components
        crop: Flag to crop the component to minimal dimensions (defaults to True)

    Returns:
        A list of measurements corresponding to the result of the function call on each component.
    """
    measurements = []
    for component in components:
        if crop:
            component = crop_image(component)
        measurement = func(component)
        measurements.append(measurement)
    return np.array(measurements)
