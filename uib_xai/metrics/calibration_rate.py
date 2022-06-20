""" Module containing metrics proposed by T. Gomez et al.

T. Gomez et al. (2019) proposed a set of metrics to evaluate the goodness of a saliency map method.
The metrics of this module allow to evaluate if the saliency map is correctly calibrated. These
metrics are calculated it with perturbation of an image depending on the respective saliency map.
There are two strategies to perturb the image:
    Deletion: The image is perturbed replacing parts of the image with a value
    Insertion: The image is completly perturbed and then restored with the original value

Abstract:
    Due to the black-box nature of deep learning models, there is a recent development of solutions
    for visual explanations of CNNs. Given the high cost of user studies, metrics are necessary to
    compare and evaluate these different methods. In this paper, we critically analyze the Deletion
    Area Under Curve (DAUC) and Insertion Area Under Curve (IAUC) metrics proposed by
    Petsiuk et al. (2018). These metrics were designed to evaluate the faithfulness of saliency maps
    generated by generic methods such as Grad-CAM or RISE. First, we show that the actual saliency
    score values given by the saliency map are ignored as only the ranking of the scores is taken
    into account. This shows that these metrics are insufficient by themselves, as the visual
    appearance of a saliency map can change significantly without the ranking of the scores being
    modified. Secondly, we argue that during the computation of DAUC and IAUC, the model is
    presented with images that are out of the training distribution which might lead to an
    unreliable behavior of the model being explained. To complement DAUC/IAUC, we propose new
    metrics that quantify the sparsity and the calibration of explanation methods, two previously
    unstudied properties. Finally, we give general remarks about the metrics studied in this paper
    and discuss how to evaluate them in a user study.

References:
    http://arxiv.org/abs/2201.13291

Writen by: Miquel Miró Nicolau (UIB)
"""
from typing import Callable, Union

import numpy as np
import torch
from scipy import stats

from . import utils

__all__ = ["deletion"]


def deletion(
    image: np.array,
    saliency_map: np.array,
    prediction_func: Callable,
    region_shape: tuple,
    perturbation_value: Union[int, Callable],
) -> float:
    """

    Args:
        image:
        saliency_map:
        prediction_func:
        region_shape:
        perturbation_value:

    Returns:

    """
    regions, regions_values = utils.get_regions(saliency_map.numpy(), region_shape)

    pre_prediction = prediction_func(image)
    original_id = np.argmax(pre_prediction)

    img_perturbed = torch.clone(image.detach())
    perturbation_scores = []

    for region in regions:
        img_perturbed = utils.perturb_img(
            img_perturbed, region, region_shape, perturbation_value
        )
        now_pred = prediction_func(img_perturbed)

        perturbation_scores.append(pre_prediction[original_id] - now_pred[original_id])
        pre_prediction = now_pred

    faith, _ = stats.pearsonr(regions_values, perturbation_scores)

    return faith
