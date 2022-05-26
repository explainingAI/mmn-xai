# -*- coding: utf-8 -*-
""" This module contains a set of function to generate images out of the domain.

Written by: Miquel Miró Nicolau (UIB), 2022
"""
from typing import Union

import numpy as np


def generate_salt_and_pepper(size: tuple, p=0.5):
    """ Generate salt and pepper noise.

    Args:
        size (Tuple): Size of the image.
        p (float): Probability of salt and pepper noise.

    Returns:
        numpy.ndarray: Salt and pepper noise.
    """
    out = np.zeros(size)
    out[np.random.rand(*size) < p] = 0
    out[np.random.rand(*size) < p] = 255
    return out


def generate_gaussian_noise(size: tuple, mean=0, std=1):
    """ Generate gaussian noise.

    Args:
        size (Tuple): Size of the image.
        mean (float): Mean of gaussian noise.
        std (float): Standard deviation of gaussian noise.

    Returns:
        numpy.ndarray: Gaussian noise.
    """
    out = np.zeros(size)
    out += np.random.normal(mean, std, size)
    return out


def generate_speckle_noise(size: tuple, mean=0, std=1):
    """ Generate speckle noise.

    Args:
        size (Tuple): Size of the image.
        mean (float): Mean of speckle noise.
        std (float): Standard deviation of speckle noise.

    Returns:
        numpy.ndarray: Speckle noise.
    """
    out = np.zeros(size)
    out += np.random.normal(mean, std, out.shape)
    out = out + np.random.normal(mean, std, out.shape)
    return out


def generate_constant_images(size, value: Union[int, float]):
    """ Generate constant images.

    Args:
        size (Tuple): Size of the image.
        value (int|float): Value to fill the constant image.

    Returns:
        numpy.ndarray: Constant images.
    """
    out = np.zeros(size)
    out += value
    return out


def generate_random_contant_images(size: tuple):
    """ Generate constant images with random values.

    Args:
        size (Tuple): Size of the image.
    """
    return generate_constant_images(size, np.random.uniform(0, 255) / 255)


def get_random_images(n_images: int, size: tuple):
    """ Generate random images.

    Args:
        n_images: Number of images to generate.
        size: Tupe with the size of the images.

    """
    for i in range(n_images // 4):

        images_funcs = [generate_random_contant_images, generate_gaussian_noise,
                        generate_speckle_noise, generate_salt_and_pepper]

        for img_f in images_funcs:
            yield (255 * img_f(size)).astype(np.uint8)
