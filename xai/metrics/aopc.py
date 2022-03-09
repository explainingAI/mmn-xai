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
from typing import List, Union, Callable

import numpy as np


def __get_regions(saliency_map, region_shape, reverse: bool = False):
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

    def __get_region_value(region, sal_map, reg_shape):
        """ Auxiliary function to sort the regions by their saliency map value.

        Args:
            region: Tuple, with the start and end points of the region.
            sal_map: 2D Numpy array with the saliency map.
            reg_shape: Tuple with the shape of the region.

        Returns:
            The saliency map value of the region.
        """
        return np.sum(
            sal_map[region[0]: region[0] + reg_shape[0], region[1]: region[1] + reg_shape[1]])

    regions = []

    for horizontal_split in range(0, saliency_map.shape[0], region_shape[0]):
        for vertical_split in range(0, saliency_map.shape[1], region_shape[1]):
            regions.append((horizontal_split, vertical_split))

    regions = sorted(regions, key=lambda x: __get_region_value(x, saliency_map, region_shape),
                     reverse=reverse)

    return regions


def __perturb_img(img, region, region_size, perturbation_value: Union[Callable, int]):
    """ Perturb the image in the given region with the given value

    The perturbation value can be a function that returns the value to be perturbed or a
    constant value. In both cases, the value is set equally in all the pixels in the region.

    Args:
        img: 2D or 3D Numpy array with the image.
        region: Sub-region of the image to be perturbed.
        region_size: Tuple with the size of the region.
        perturbation_value: Value to be perturbed in the region. Function or constant.

    Returns:
        The perturbed image.
    """
    if callable(perturbation_value):
        perturbation_value = perturbation_value()

    img_copy = np.copy(img)

    if len(img_copy.shape) > 2:
        img_copy[region[0]: region[0] + region_size[0],
                 region[1]: region[1] + region_size[1], :] = perturbation_value
    else:
        img_copy[region[0]: region[0] + region_size[0],
                 region[1]: region[1] + region_size[1]] = perturbation_value
    return img_copy


def aopc(dataset: List[np.ndarray], saliency_maps: List[np.ndarray], prediction_func, region_shape,
         value, reverse: bool = False):
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
        original_id = np.argmax(original_prediction)  # llevar

        img_perturbed = np.copy(img)
        regions = __get_regions(sal_map, region_shape=region_shape, reverse=reverse)

        for reg in regions:
            img_perturbed = __perturb_img(img_perturbed, reg, region_shape,
                                          perturbation_value=value)

            perturbed_prediction = prediction_func(img_perturbed)
            aopc_value += (original_prediction[original_id] - perturbed_prediction[original_id])

    if len(regions) < 0:
        raise ValueError("Must always exist at least a region")

    aopc_value /= len(dataset)
    aopc_value /= len(regions)

    return aopc_value
