# -*- coding: utf-8 -*-
""" Module for calculating sparcity of a matrix.

Sparcity is proposed by T. Gomez et al. as an auxiliary metric for evaluating the quality of a
saliency map. In the original paper is defined as S=Smax / Smean, where Smax is the maximum value
of the saliency map and Smean is the mean value of the saliency map. This definition is converted
to S=1/Smean, when the saliency map is normalized between 0 and 1.


References:
    http://arxiv.org/abs/2201.13291

Written by: Miquel Miró Nicolau (UIB)
"""
import numpy as np

from . import utils


def sparcity(saliency_map: np.ndarray):
    sal_map_n = utils.normalize_zero_one(saliency_map)

    mean = np.mean(sal_map_n)

    sparc = 1 / mean if mean > 0 else mean

    return sparc
