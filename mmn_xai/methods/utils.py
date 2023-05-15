""" This module contains utility functions for the XAI methods.

Written by: Miquel Miró Nicolau (UIB), 2023
"""
from typing import Callable

import cv2
import numpy as np
import torch


def to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def normalize(expl: np.ndarray) -> np.ndarray:
    """Normalize the explanation values to the range [0, 1].

    Args:
        expl: numpy array of shape (H, W) containing the explanation values

    Returns:
        numpy array of shape (H, W) containing the normalized explanation values
    """
    expl = abs(expl)
    expl = expl - expl.min()
    expl = expl / expl.max()

    return expl


def densify(
    expl: np.ndarray, image: np.ndarray, agg_func: Callable, norm: bool = True
) -> np.ndarray:
    """Densify the explanation by aggregating the explanation values of the pixels belonging to
    the same object.

    The saliency maps obtained by the XAI methods can be sparse or dense. A sparse saliency map
    is a map where the explanations values are not uniform across an object, but intead they are
    separated in different pixels with pixels of value 0 in between. A dense saliency map is a map
    where the explanation values are uniform across an object. LIME saliency map is an example of
    a dense saliency map, while the saliency maps obtained by LRP are examples of sparse saliency
    maps.

    Args:
        expl: numpy array of shape (H, W) containing the explanation values
        image: numpy array of shape (H, W) containing the image
        agg_func: aggregation function to be applied to the explanation values of the pixels
                belonging to the same object
        norm: Boolean indicating whether the explanation values should be normalized to the
                range [0, 1]

    Returns:
        numpy array of shape (H, W) containing the densified explanation
    """
    if expl.shape != image.shape:
        raise ValueError(f"Different shapes {expl.shape} != {image.shape}")

    mask = np.copy(image)

    if mask.max() > 1:
        mask = mask / mask.max()

    mask[mask != 1] = 0
    mask = mask.astype(np.uint8)

    dense_expl = []
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        mask_c = np.zeros_like(image).astype(np.float32)
        mask_c = cv2.drawContours(
            image=mask_c, contours=[c], contourIdx=-1, color=1, thickness=-1
        )
        mask_c[mask_c == 1] = agg_func(expl[mask_c == 1])

        dense_expl.append(mask_c)

    dense_expl = np.array(dense_expl)
    dense_expl = np.sum(dense_expl, axis=0)

    dense_expl += expl * (dense_expl == 0).astype(np.uint8)

    if norm:
        dense_expl[dense_expl != 0] = dense_expl[dense_expl != 0] / np.max(dense_expl)

    return dense_expl


def get_activation(image, layer, model):
    activation = {}

    def hook(model, input, output):
        activation["layer"] = output.detach()

    layer.register_forward_hook(hook)
    _ = model(image)
    layer_output = activation["layer"]

    return layer_output.detach()
