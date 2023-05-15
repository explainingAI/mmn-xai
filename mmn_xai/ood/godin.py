# flake8: noqa
# pylint: skip-file
""" Module containing the implementation of the GODIN method.

This module contains two implementation of GODIN method. The first one is a PyTorch implementation
and the second one is a Tensorflow implementation.

References:
    - https://arxiv.org/abs/2002.11297

Written by: Miquel Miró Nicolau (UIB), 2023
"""
import tensorflow as tf
import torch
from tensorflow.keras import layers as tf_layers
from tensorflow.keras.regularizers import l2
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


def tf_godin(x, n_classes=10, weight_decay: float = 0.0005):
    """Tensorflow classifier

    Args:
        x: input into the classifier
        n_classes: number of classes
        weight_decay:

    Returns:
        The output of GODIN
    """
    # Pool the feature maps after the end of all the residual blocks
    x = tf_layers.BatchNormalization()(x)
    x = tf_layers.ReLU()(x)
    x = tf_layers.AveragePooling2D(pool_size=8)(x)

    # Flatten into 1D vector
    x = tf_layers.GlobalAvgPool2D()(x)

    # Define the ODIN as specified in Section 3.1.1 of
    # https://arxiv.org/abs/2002.11297
    h = tf_layers.Dense(n_classes, kernel_initializer="he_normal")(x)

    g = tf_layers.Dense(1, kernel_regularizer=l2(weight_decay))(x)
    g = tf_layers.BatchNormalization()(g)
    g = tf_layers.Activation("sigmoid")(g)
    outputs = tf.math.divide(h, g)

    return outputs
