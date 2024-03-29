""" XAI Metrics: Faithfullness.

This module contains the Faithfullness implementation metric.

The faithfulness Fx for a single image x is calculated by taking the Pearson correlation between the
relevance Ri assigned to pixel i by the saliency method, and the change in classification output
when pixel i is perturbed  to create image xi: Δi = f(x) − f(xi).

References: https://aaai.org/ojs/index.php/AAAI/article/view/6064
Writen by: Miquel Miró Nicolau (UIB)
"""
from typing import Callable, Union

import numpy as np
import torch
from scipy import stats

from . import utils

__all__ = ["faithfullness"]


def faithfullness(
    img: np.ndarray,
    saliency_map: np.ndarray,
    prediction_func: Callable,
    region_shape: tuple,
    value: Union[Callable, int, float],
) -> Union[float, int]:
    """Main function for the calculation of the faithfullness metric.

    Args:
        img: NumPy array with the image.
        saliency_map: NumPy array with the saliency map.
        prediction_func: Function that predicts the class of an image.
        region_shape: Shape of the region to be considered.
        value: Value or method to be used for the calculation of the perturbation.

    Returns:
        The calculated faithfullness metric.
    """
    regions, regions_values = utils.get_regions(saliency_map, region_shape)

    original_prediction = prediction_func(img)
    original_idx = np.argmax(original_prediction)
    perturb_preds = []

    for region in regions:
        perturbed_img = torch.clone(img.detach())
        perturbed_img = utils.perturb_img(perturbed_img, region, region_shape, value)
        prediction_pert = prediction_func(perturbed_img)

        perturb_preds.append(
            original_prediction[original_idx] - prediction_pert[original_idx]
        )

    regions_values, perturb_preds = np.array(regions_values), np.array(perturb_preds)
    perturb_preds = np.squeeze(perturb_preds)

    max_regions = regions_values.max() if regions_values.max() > 0 else 1
    max_perturb = perturb_preds.max() if perturb_preds.max() > 0 else 1

    regions_values = regions_values / max_regions
    perturb_preds = perturb_preds / max_perturb

    faith, _ = stats.pearsonr(regions_values, perturb_preds)

    return faith
