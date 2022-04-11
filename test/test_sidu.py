# -*- coding: utf-8 -*-
"""

"""
import unittest

import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from uib_xai.methods import sidu


class Layer:

    def __init__(self, weights):
        self.__weights = weights

    @property
    def weights(self):
        return self.__weights

    @property
    def shape(self):
        return self.__weights.shape

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self


class ModelTest:

    def __init__(self, weights):
        self.__layer = Layer(weights)

    def __call__(self, *args, **kwargs):
        return torch.ones((1, 2))

    def __getitem__(self, item):
        return self.__layer


def initialize_model(num_classes=2, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    def set_parameter_requires_grad(model, feature_extracting):
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

    def test_color_all_zeros(self):
        weights = torch.ones((1, 10, 5, 5))
        model = ModelTest(weights)

        image = torch.zeros((1, 3, 50, 50))
        importance = sidu.sidu(model, weights, image=image)

        self.assertAlmostEqual(np.count_nonzero(importance.cpu().numpy()), 0)

    def test_bw_all_zeros(self):
        weights = torch.ones((1, 10, 5, 5))
        model = ModelTest(weights)

        image = torch.zeros((1, 1, 50, 50))
        importance = sidu.sidu(model, weights, image=image)

        self.assertAlmostEqual(np.count_nonzero(importance.cpu().numpy()), 0)

    # def test_real_model(self):
    #     model_ft, input_size = initialize_model()
    #     img_stub = torch.zeros((1, 3, 512, 512))
    #
    #     activation = {}
    #
    #     def get_activation(name):
    #         def hook(model, input, output):
    #             activation[name] = output.detach()
    #
    #         return hook
    #
    #     model_ft.layer4[1].conv2.register_forward_hook(get_activation('layer4_1_conv2'))
    #     _ = model_ft(img_stub)
    #
    #     sidu.sidu(model_ft, activation['layer4_1_conv2'], img_stub)
