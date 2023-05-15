""" Suite of tests for the lrpCAM.

Writen by: Miquel Miró-Nicolau (UIB), 2023.
"""
import unittest

import torch

from mmn_xai.methods import lrp_cam


class TestModel(unittest.TestCase):
    def test_cnn_captum(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=2, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=200, out_features=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2, out_features=1),
        )

        sub_model = torch.nn.Sequential(*list(model)[2:])

        xai = lrpCAM.lrp_cam(model, sub_model)
        expl = xai(image=torch.zeros((1, 1, 10, 10)), target=None, layer=model[0])

        self.assertTupleEqual(tuple(expl.shape), (10, 10))
