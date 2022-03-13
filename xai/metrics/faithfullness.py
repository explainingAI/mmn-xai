# -*- coding: utf-8 -*-
""" XAI Metrics: Faithfullness.

This module contains the Faithfullness implementation metric.

The faithfulness Fx for a single image x is calculated by taking the Pearson correlation between the
relevance Ri assigned to pixel i by the saliency method, and the change in classification output
when pixel i is perturbed  to create image xi: Δi = f(x) − f(xi).

References: https://aaai.org/ojs/index.php/AAAI/article/view/6064
Writen by: Miquel Miró Nicolau (UIB)
"""
from typing import Union

from scipy import stats
import numpy as np
import torch

from . import utils


def faithfullness(img, saliency_map, prediction_func, region_shape, value) -> Union[float, int]:
    """ Main function for the calculation of the faithfullness metric.

    Args:
        img: NumPy array with the image.
        saliency_map: NumPy array with the saliency map.
        prediction_func: Function that predicts the class of an image.
        region_shape: Shape of the region to be considered.
        value: Value or method to be used for the calculation of the perturbation.

    Returns:
        The calculated faithfullness metric.
    """
    regions, regions_values = utils.get_regions(saliency_map.numpy(), region_shape)

    original_prediction = prediction_func(img)
    original_idx = torch.argmax(original_prediction)
    perturb_preds = []

    perturbed_img = img
    for region in regions:
        perturbed_img = utils.perturb_img(perturbed_img, region, region_shape, value)
        prediction_pert = prediction_func(perturbed_img)

        perturb_preds.append(
            (original_prediction[original_idx] - prediction_pert[original_idx]).numpy())

    regions_values, perturb_preds = np.array(regions_values), np.array(perturb_preds)
    perturb_preds = np.squeeze(perturb_preds)
    regions_values = utils.normalize_zero_one(regions_values)
    perturb_preds = utils.normalize_zero_one(perturb_preds)

    faith, _ = stats.pearsonr(regions_values, perturb_preds)

    return faith
