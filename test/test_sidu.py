# -*- coding: utf-8 -*-
"""

"""
import unittest

import numpy as np
import torch

from uib_xai.methods import sidu


class Layer:

    def __init__(self, weights):
        self.__weights = weights

    @property
    def weights(self):
        return self.__weights


class TestModel:

    def __init__(self, weights):
        self.__layer = Layer(weights)

    def __call__(self, *args, **kwargs):
        return torch.ones((1, 2))

    def __getitem__(self, item):
        return self.__layer


class TestSIDU(unittest.TestCase):

    def test_color_all_zeros(self):
        weights = torch.ones((5, 5, 5))
        model = TestModel(weights)

        image = torch.zeros((50, 50, 3))
        importance = sidu.sidu(model, Layer(weights), image=image)

        self.assertAlmostEqual(np.count_nonzero(importance), 0)

    def test_bw_all_zeros(self):
        weights = torch.ones((5, 5, 5))
        model = TestModel(weights)

        image = torch.zeros((50, 50))
        importance = sidu.sidu(model, Layer(weights), image=image)

        self.assertAlmostEqual(np.count_nonzero(importance), 0)
