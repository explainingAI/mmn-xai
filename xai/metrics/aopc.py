# -*- coding: utf-8 -*-
""" AOPC (Average Order of Precision Classification) metric.

Abstract of the original paper:
    Deep neural networks (DNNs) have demonstrated impressive performance in complex machine learning
    tasks such as image classification or speech recognition. However, due to their multilayer
    nonlinear structure, they are not transparent, i.e., it is hard to grasp what makes them arrive
    at a particular classification or recognition decision, given a new unseen data sample.
    Recently, several approaches have been proposed enabling one to understand and interpret the
    reasoning embodied in a DNN for a single test image. These methods quantify the “importance” of
    individual pixels with respect to the classification decision and allow a visualization in terms
    of a heatmap in pixel/input space. While the usefulness of heatmaps can be judged subjectively
    by a human, an objective quality measure is missing. In this paper, we present a general
    methodology based on region perturbation for evaluating ordered collections of pixels such as
    heatmaps. We compare heatmaps computed by three different methods on the SUN397, ILSVRC2012,
    and MIT Places data sets. Our main result is that the recently proposed layer-wise relevance
    propagation algorithm qualitatively and quantitatively provides a better explanation of what
    made a DNN arrive at a particular classification decision than the sensitivity-based approach or
    the deconvolution method. We provide theoretical arguments to explain this result and discuss
    its practical implications. Finally, we investigate the use of heatmaps for unsupervised
    assessment of the neural network performance.

Refs:
     https://ieeexplore.ieee.org/document/7552539

Authors:
    Samek, Wojciech; Binder, Alexander; Montavon, Gregoire; Lapuschkin, Sebastian; Muller, Klaus-Robert


Writen by: Miquel Miró Nicolau (UIB)
"""
import torch

from . import utils


def aopc(dataset, saliency_maps, prediction_func, region_shape, value, reverse: bool = False):
    """ Approximate Optimal Perturbation Criterion.

    The AOPC metric is a measure of the quality of the adversarial perturbation. The metric is
    calculated as the difference between the original prediction and the prediction of the perturbed
    image. The perturbation depends on the saliency map of the image.

    Args:
        dataset: List of images to obtain the AOPC.
        saliency_maps: List of saliency maps for each image in the dataset.
        prediction_func: Function that returns the prediction of the model for a given image.
        region_shape: Tuple with the shape of the region.
        value: Value to be perturbed in the region. Function or constant.
        reverse: Boolean that indicates if the regions should be sorted in descending order.

    Returns:
        Numeric value with the AOPC for the given dataset.
    """

    if len(dataset) != len(saliency_maps):
        raise ValueError("Must pass the same number of images and saliency maps")

    aopc_value = 0
    regions = []
    for img, sal_map in zip(dataset, saliency_maps):
        original_prediction = prediction_func(img)
        original_id = torch.argmax(original_prediction)  # llevar

        img_perturbed = torch.clone(img)
        regions, _ = utils.get_regions(sal_map, region_shape=region_shape, reverse=reverse)

        for reg in regions:
            img_perturbed = utils.perturb_img(img_perturbed, reg, region_shape,
                                              perturbation_value=value)

            perturbed_prediction = prediction_func(img_perturbed)
            aopc_value += (original_prediction[original_id] - perturbed_prediction[original_id])

    if len(regions) < 0:
        raise ValueError("Must always exist at least a region")

    aopc_value /= len(dataset)
    aopc_value /= len(regions)

    return aopc_value
