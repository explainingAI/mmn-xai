""" Suite of tests for the lrpCAM.

Writen by: Miquel Miró-Nicolau (UIB), 2023.
"""
import unittest

import torch

from mmn_xai.methods import w_smooth_grad


class TestModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.__model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=2, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=180, out_features=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2, out_features=1),
        )
        super().__init__(*args, **kwargs)

    def test_cnn_single_batch(self):
        xai = w_smooth_grad.WeightedSmoothGrad(self.__model)

        expl = xai(torch.zeros((1, 1, 9, 10)))

        self.assertTupleEqual(tuple(expl.shape), (1, 1, 9, 10))

    def test_cnn_batch_mult(self):
        xai = w_smooth_grad.WeightedSmoothGrad(self.__model)

        expl = xai(torch.zeros((8, 1, 9, 10)))

        self.assertTupleEqual(tuple(expl.shape), (8, 1, 9, 10))

