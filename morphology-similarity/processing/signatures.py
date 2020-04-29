"""
Contains signature extraction functions.
"""
import numpy as np
from skimage import measure


def pixel_ratio_sig(component):
    """The ratio of white pixels to black pixels"""
    # Note: We are guranteed to have at least 1 pixel of value 1
    return np.sum(component == 0)/np.sum(component == 1)


def shape_ratio_sig(component):
    """The ratio of the width of the component to it's height"""
    return component.shape[0]/component.shape[1]


def surface_volume_ratio_sig(component):
    """The surface to volume (perimeter to area) ratio"""
    perimeter = measure.perimeter(component)
    area = np.sum(component == 1)
    # Note: We are guranteed to have at least 1 pixel of value 1
    return perimeter/area
