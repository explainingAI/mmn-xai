""" Module containing the functions to explain nn with LIME.

Written by: Miquel Miró Nicolau (2023), UIB
"""
import copy
from typing import Callable

import cv2
import numpy as np
import torch
from lime import lime_image


def swap_colors(pixels):
    """Function to perturb the image by swapping the colors of the pixels.


    Args:
        pixels: Numpy array of shape (n, m) with the pixel values.

    Returns:
        Numpy images perturbed.
    """
    mean = np.mean(pixels)
    ret_val = ((mean + 1) % 2) + 1

    return ret_val


def batch_predict(image: np.array, network: Callable, multi_channel: bool) -> np.array:
    """Function to predict the output of the network for a batch of images.

    Args:
        image: NumPy array of shape (n, m, 3) with the image.
        network: Callable function to predict the output of the network.

    Returns:
        NumPy array with the output of the network.
    """
    image = np.copy(image)

    image = np.transpose(image, (0, 3, 1, 2))

    if not multi_channel:
        image = image[:, 0:1, :, :]
    logit = network(image).reshape((-1, 1))

    return logit


def defined(
    image: np.array, attribution_function: Callable, g_x: Callable = None
) -> np.array:
    """Predict using the attribution function directly

    Args:
        image: Numpy array of shape (b, c, n, m) with a batch of images.
        attribution_function: Callable function to predict the attribution.
        g_x: Function to extract visual patterns from the images

    Returns:
        Numpy array of shape 1 with the result.
    """
    if g_x is None:
        g_x = count_by_value

    image = image[0, 0, :, :]

    res = g_x(image)
    res = attribution_function([res[1], res[2], res[3]])

    return res


def count_by_value(image: np.array) -> np.array:
    """Function to count the number of pixels with each value.

    Counts the number of objects with each different value.

    Args:
        image: Numpy array of shape (n, m) with the image.

    Returns:
        Numpy array with the number of objects with each different value.
    """
    image = np.copy(image)
    res = {1: 0, 2: 0, 3: 0}

    for val in np.unique(image)[1:]:
        aux_img = (image == val).astype(np.uint8) * 255

        contours, _ = cv2.findContours(aux_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        res[val] = len(contours)

    return res


def own_segment(image: np.array) -> np.array:
    """Function to segment the image into different objects.

    LIME methods needs a segmented images. This image is used to perturb the original image. This
    method allows to identify the different objects according to their value and separates them
    into different classes. Each different class is represented with an increasing value from 1 to
    N.

    Args:
        image: 3-Channel numpy array containing the image to segment.

    Returns:
        2-Channel images where the values of pixel indicades the class that represents.
    """
    img_seg = np.zeros_like(image.astype(np.uint8)[:, :, 0], dtype=np.uint8)
    image = copy.deepcopy(image)
    image[image >= 1] = 1

    cont, _ = cv2.findContours(
        image.astype(np.uint8)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for i, c in enumerate(cont):
        cv2.drawContours(img_seg, [c], -1, i + 1, -1)

    return img_seg


def get_exp(
    explainer: lime_image.LimeImageExplainer,
    img: np.array,
    net: Callable,
    device: str,
    hide_color_fn=None,
    segmentation_fn=None,
    num_samples: int = 1500,
    batch_size: int = None,
):
    """Wrapper for the LIME method.

    Args:
        explainer: Object of the LIME library to explain the image.
        img: Numpy array containing the image to explain.
        net: Callable object,
        device: String with the device to use.
        multi_channel: Boolean to indicate if the image has multiple channels.
        hide_color_fn: Function to perturb the image.
        segmentation_fn: Function to segment the image.
        num_samples: Integer, parameter of LIME method.
        batch_size: Integer, batch size to use.
    Returns:
        Numpy array with the explanation.
    """
    if batch_size is None:
        batch_size = img.shape[0]
    lime_res = []
    explanation = explainer.explain_instance(
        img,
        lambda x: batch_predict(
            x,
            lambda x: net(torch.from_numpy(x.astype(np.float32)).to(device))
            .detach()
            .cpu()
            .numpy(),
            multi_channel=x.shape[1] > 1,
        ),
        top_labels=1,
        hide_color=hide_color_fn,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
        batch_size=batch_size,
        random_seed=42,
        progress_bar=False,
    )

    mask = np.zeros(
        (explanation.segments.shape[0], explanation.segments.shape[1], 3),
        dtype=np.float64,
    )

    for k in explanation.local_exp.keys():
        for key, val in explanation.local_exp[k]:
            if key != 0:
                mask[:, :, k][explanation.segments == key] = abs(val)
    lime_res.append(mask)
    lime_res = np.array(lime_res)

    return lime_res[:, :, :, 0]
