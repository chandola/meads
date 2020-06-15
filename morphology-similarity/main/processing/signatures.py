"""
Contains signature extraction functions.
"""
import numpy as np
from skimage import measure
from .utils import crop_image, get_inscribed_rect_area


def pixel_ratio_sig(component):
    """The ratio of white pixels to black pixels"""
    # Note: We are guranteed to have at least 1 pixel of value 1
    return np.sum(component == 0)/np.sum(component == 1)


def shape_ratio_sig(component):
    """The ratio of the width of the component to it's height"""
    return component.shape[0]/component.shape[1]


def perimeter_area_ratio_sig(component):
    """The surface to volume (perimeter to area) ratio"""
    perimeter = measure.perimeter(component)
    area = np.sum(component == 1)
    # Note: We are guranteed to have at least 1 pixel of value 1
    return perimeter/area


def surface_volume_ratio_sig(component):
    """Alias for perimeter to area ratio"""
    return perimeter_area_ratio_sig(component)


def rect_area_ratio_sig(component):
    """
    The ratio of the area of the largest inscribed rectangle
    to the area of the smallest circumscribed rectangle
    """
    cropped_component = crop_image(component)
    circ_area = cropped_component.shape[0] * cropped_component.shape[1]
    insc_area = get_inscribed_rect_area(cropped_component)[0]
    return insc_area/circ_area


def average_pixel_intensity_sig(component):
    """The average pixel value"""
    return np.mean(component)
