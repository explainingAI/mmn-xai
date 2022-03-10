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


def sparcity(saliency_map: np.ndarray):
    sal_map_n = np.copy(saliency_map)

    sal_map_n = (sal_map_n - sal_map_n.min()) / (sal_map_n.max() - sal_map_n.min())

    return 1 / np.mean(sal_map_n)
