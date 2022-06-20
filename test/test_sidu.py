""" Set of tests for SIDU method proposed by Muddamsetty et al. (2021)

Written by: Miquel Miro-Nicolau (UIB) 2022
"""
import unittest
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from uib_xai.methods import sidu


class Layer:
    def __init__(self, weights: Union[torch.Tensor, np.array]):
        self.__weights = weights

    @property
    def weights(self) -> Union[list, torch.Tensor]:
        return self.__weights

    @property
    def shape(self) -> tuple:
        return self.__weights.shape

    def cpu(self) -> object:
        return self

    def detach(self) -> object:
        return self

    def numpy(self) -> object:
        return self


class ModelTest:
    def __init__(self, weights: np.array):
        self.__layer = Layer(weights)

    def __call__(self, data: np.array, *args: list, **kwargs: dict) -> torch.Tensor:
        return torch.ones((data.shape[0], 2))

    def __getitem__(self, item: int) -> Layer:
        return self.__layer


def initialize_model(
    num_classes: int = 2, feature_extract: bool = True, use_pretrained: bool = True
) -> Tuple[torch.nn.Module, int]:
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    def set_parameter_requires_grad(
        model: torch.nn.Module, feature_extracting: bool
    ) -> None:
        if feature_extracting:
            for param in list(model.parameters())[:30]:
                param.requires_grad = False

    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size


class TestSIDU(unittest.TestCase):
    def test_color_all_zeros(self) -> None:
        weights = torch.ones((1, 10, 5, 5))
        model = ModelTest(weights)

        image = torch.zeros((1, 3, 50, 50))
        importance = sidu.sidu(model, weights, image=image)

        self.assertAlmostEqual(np.count_nonzero(importance.cpu().numpy()), 0)

    def test_bw_all_zeros(self) -> None:
        weights = torch.ones((1, 10, 5, 5))
        model = ModelTest(weights)

        image = torch.zeros((1, 1, 50, 50))
        importance = sidu.sidu(model, weights, image=image)

        self.assertAlmostEqual(np.count_nonzero(importance.cpu().numpy()), 0)
