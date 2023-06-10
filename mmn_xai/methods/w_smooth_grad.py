""" Weighted SmoothGrad.

Ref: https://github.com/pkmr06/pytorch-smoothgrad/blob/master/lib/gradients.py
"""
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from mmn_xai.methods import gradient_methods


class WeightedSmoothGrad(gradient_methods.GradientMethod):
    """ Weighted SmoothGrad.

    New method based on the SmoothGrad method. It adds a weight to the gradients when combining
    them.
    """

    def __init__(
            self,
            pretrained_model,
            device="cpu",
            st_dev_spread=0.15,
            n_samples=25,
            magnitude=True,
            add_softmax: bool = False,
            distance: Optional[Callable] = None
    ):
        super().__init__(pretrained_model, device, add_softmax)

        self.st_dev_spread = st_dev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

        if distance is None:
            distance = lambda x, y: 1 - min(abs((x - y).pow(2).sum().sqrt()), 1)
        self.distance = distance

    def __call__(self, x, index=None):
        output, grad = self.output_and_gradient(x.to(self.device))
        org_output = output

        if self.add_softmax:
            org_output = nn.functional.softmax(org_output)

        x = x.data.cpu().numpy()
        stdev = self.st_dev_spread * (np.max(x) - np.min(x))

        total_gradients = np.zeros_like(x)
        for _ in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            output, grad = self.output_and_gradient(x_plus_noise)
            if self.add_softmax:
                output = nn.functional.softmax(output)

            dist = self.distance(org_output, output)
            if isinstance(dist, torch.Tensor):
                dist = dist.detach().cpu().numpy()

            grad = dist * grad
            if self.magnitude:
                grad = grad * grad

            if not np.isnan(grad).any():
                total_gradients += grad

        avg_gradients = total_gradients[0, :, :, :] / float(self.n_samples)
        if avg_gradients.max() != 0:
            avg_gradients /= avg_gradients.max()

        return avg_gradients
