# -*- coding: utf-8 -*-
""" Utilities for metrics.

Written by: Miquel Miró Nicolau (UIB)
"""

from typing import Union, Callable

import numpy as np
import torch


def normalize_zero_one(data):
    """ Normalize the data to be between 0 and 1.

    Args:
        data: NumPy array with the data to be normalized.

    Returns:
        Numpy array with the normalized data.
    """
    data_normalized = np.copy(data)
    data_normalized = (data_normalized - data_normalized.min()) / \
                      (data_normalized.max() - data_normalized.min())

    return data_normalized


def get_regions(saliency_map, region_shape, reverse: bool = False):
    """ Get the regions of the saliency map that will be perturbed

    The regions are obtained by sliding the region_shape over the saliency map. Then are sorted
    by the addition of the saliency map values in the region.

    Args:
        saliency_map: 2D Numpy array with the saliency map.
        region_shape: Tuple with the shape of the region.
        reverse: Boolean that indicates if the regions should be sorted in descending order.

    Returns:
        List of tuples with the regions.
    """

    regions = []
    regions_values = []

    for horizontal_split in range(0, saliency_map.shape[0], region_shape[0]):
        for vertical_split in range(0, saliency_map.shape[1], region_shape[1]):
            regions.append((horizontal_split, vertical_split))
            regions_values.append(np.sum(
                saliency_map[horizontal_split: horizontal_split + region_shape[0],
                vertical_split: vertical_split + region_shape[1]]))

    regions = sorted(zip(regions, regions_values), key=lambda x: x[1], reverse=reverse)

    return zip(*regions)


def perturb_img(img, region, region_size, perturbation_value: Union[Callable, int]):
    """ Perturb the image in the given region with the given value

    The perturbation value can be a function that returns the value to be perturbed or a
    constant value. In both cases, the value is set equally in all the pixels in the region.

    Args:
        img: Pytorch tensor with the image.
        region: Sub-region of the image to be perturbed.
        region_size: Tuple with the size of the region.
        perturbation_value: Value to be perturbed in the region. Function or constant.

    Returns:
        The perturbed image.
    """
    if callable(perturbation_value):
        perturbation_value = perturbation_value()

    img_copy = torch.clone(img)

    if len(img_copy.shape) > 2:
        img_copy[region[0]: region[0] + region_size[0],
        region[1]: region[1] + region_size[1], :] = perturbation_value
    else:
        img_copy[region[0]: region[0] + region_size[0],
        region[1]: region[1] + region_size[1]] = perturbation_value
    return img_copy
