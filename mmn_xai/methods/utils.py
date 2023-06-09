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


def raw_densify(
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


from numpy.lib.stride_tricks import as_strided


def pool2d(data: np.ndarray, kernel_size: int, stride: int, padding: int = 0):
    """Method that applies the pooling operation over an image.

    Args:
        data:
        kernel_size:
        stride:
        padding:

    Returns:

    """
    data = np.pad(data, padding, mode="constant")

    output_shape = (
        (data.shape[0] - kernel_size) // stride + 1,
        (data.shape[1] - kernel_size) // stride + 1,
    )

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (
        stride * data.strides[0],
        stride * data.strides[1],
        data.strides[0],
        data.strides[1],
    )

    result = as_strided(data, shape_w, strides_w)

    return result.max(axis=(2, 3))


def smooth(img: np.ndarray):
    """Applies to an image the max pooling 2D.

    Args:
        img: Numpy array with the image to smooth.

    Returns:
        Numpy array with the smoothed image.
    """
    return [
        pool2d(np.abs(img[i, :, :]), kernel_size=5, stride=1, padding=2)
        for i in range(img.shape[0])
    ]


def densify(expl, img: torch.Tensor, func: callable, *args, **kwargs):
    results = []

    for i in range(expl.shape[0]):
        expl = raw_densify(expl, img[i, 0, :, :].detach().cpu().numpy(), func, False)

        results.append(expl)

    return results
