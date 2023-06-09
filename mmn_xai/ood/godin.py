""" Module containing the implementation of the GODIN method.

This module contains two implementation of GODIN method. The first one is a PyTorch implementation
and the second one is a Tensorflow implementation.

References:
    - https://arxiv.org/abs/2002.11297

Written by: Miquel Miró Nicolau (UIB), 2023
"""
import torch
from torch import nn


class GeneralizedOdin(nn.Module):
    """Pytorch implementation of G-ODIN

    Args:
        input_size: Size of the input, number of channels of the feature map.
        output_size: Size of the output, number of classes.

    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(input_size)
        self.relu1 = nn.ReLU()
        self.avg_pool2d = nn.AvgPool2d(8)

        self.global_avg_pool2d = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # sortida igual a (1,1) == global avg

        self.h = nn.Linear(in_features=1, out_features=output_size)

        self.g = nn.Linear(in_features=1, out_features=1)
        self.g_bn = nn.BatchNorm1d(1)
        self.g_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avg_pool2d(x)
        x = self.global_avg_pool2d(x)
        x = torch.flatten(x, 1)

        h = self.h(x)
        g = self.g(x)
        g = self.g_bn(g)
        g = self.g_sigmoid(g)

        out = torch.div(h, g)

        return out


class OODModule(nn.Module):
    """ Pytorch implementation of the OODNet

    """

    def __init__(self, classes, in_features, do_sigmoid: bool = False):
        super().__init__()

        self.h = nn.Linear(in_features=in_features, out_features=classes, bias=False)
        self.g_fc = nn.Linear(in_features=in_features, out_features=1)
        self.g_bn = nn.BatchNorm1d(1)
        self.g_sigmoid = nn.Sigmoid()

        self.end_sigmoid = None

        if do_sigmoid:
            if classes > 1:
                self.end_sigmoid = nn.Softmax()
            else:
                self.end_sigmoid = nn.Sigmoid()

    def __call__(self, x, *args, **kwargs):
        h = self.h(x)

        g = self.g_fc(x)
        g = self.g_bn(g)
        g = torch.square(g)
        g = self.g_sigmoid(g)
        x = h / g

        if self.end_sigmoid is not None:
            x = self.end_sigmoid(x)

        return x, h, g
