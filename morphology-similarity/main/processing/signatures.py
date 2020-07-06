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


def perimeter_sig(component, max_peri=100):
    component = crop_image(component)
    perimeter = measure.perimeter(component)/max_peri
    return np.min([perimeter, 1.])


def area_sig(component, max_area=1000):
    """The area of the component"""
    component = crop_image(component)
    area = (component.shape[0]*component.shape[1])/max_area
    return np.min([area, 1.])


def length_sig(component, max_length=100):
    """The length of the component"""
    component = crop_image(component)
    return component.shape[0]/max_length


def height_sig(component, max_height=400):
    """The length of the component"""
    component = crop_image(component)
    return component.shape[1]/max_height


def y_sym_sig(component):
    component = crop_image(component)
    height, width = component.shape
    count = 0
    for y in range(height):
        for x in range(width//2+1):
            count += component[y, x] == component[y, width-x-1]
    return count/(height * width/2)


def x_sym_sig(component):
    component = crop_image(component)
    height, width = component.shape
    count = 0
    for y in range(height//2+1):
        for x in range(width):
            count += component[y, x] == component[height-y-1, x]
    return count/(height * width/2)


def perimeter_area_ratio_sig(component):
    """The surface to volume (perimeter to area) ratio"""
    perimeter = measure.perimeter(component)
    area = np.sum(component == 1)
    # Note: We are guranteed to have at least 1 pixel of value 1
    return perimeter/area


def surface_volume_ratio_sig(component):
    """Alias for perimeter to area ratio"""
    return perimeter_area_ratio_sig(component)


def euler_number_sig(component):
    labeled_sample = measure.label(component)
    props = measure.regionprops(labeled_sample)[0]
    return props.euler_number


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
