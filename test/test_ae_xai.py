import unittest

import torch
import numpy as np

from mmn_xai.methods import ae_xai


class TestModel(unittest.TestCase):
    def test_cnn_captum(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, out_channels=2, kernel_size=(3, 3), padding="same"
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=100, out_features=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2, out_features=1),
        )


        xai = ae_xai.AEXAI(
            5,
            model,
            (1, 10, 10),
            2
        )
        image = torch.zeros((1, 10, 10), dtype=torch.float32)
        image[:, 0:5, 0:5] = 1.0
        expl = xai(image=image)

        # self.assertTupleEqual(tuple(expl.shape), (10, 10))
