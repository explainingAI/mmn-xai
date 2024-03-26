""" Weighted SmoothGrad.

Ref: https://github.com/pkmr06/pytorch-smoothgrad/blob/master/lib/gradients.py
"""
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from mmn_xai.methods import gradient_methods


class WeightedSmoothGrad(gradient_methods.GradientMethod):
    """Weighted SmoothGrad.

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
            distance: Optional[Callable] = None,
            pot=1
    ):
        super().__init__(pretrained_model, device, add_softmax)

        self.st_dev_spread = st_dev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        self.pot = pot

        if distance is None:
            distance = lambda x, y: nn.functional.cosine_similarity(x, y, dim=0)
        self.distance = distance

    def __call__(self, x, index=None):
        output, grad = self.output_and_gradient(x.to(self.device))
        org_output = output

        if self.add_softmax:
            org_output = nn.functional.softmax(org_output)

        total_gradients = torch.zeros_like(x, device=self.device)

        x = x.data.cpu().numpy()
        stdev = self.st_dev_spread * (np.max(x) - np.min(x))

        for _ in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            output, grad = self.output_and_gradient(x_plus_noise)
            if self.add_softmax:
                output = nn.functional.softmax(output)

            dist = self.distance(org_output, output)

            # Add the same amount of dimensions than grad to allow the product
            aux = [-1] + ([1] * (len(grad.shape) - 1))
            dist = dist.reshape(aux) ** self.pot

            grad = dist * grad
            if self.magnitude:
                grad = grad * grad

            if not torch.isnan(grad).any():
                total_gradients += grad

        avg_gradients = total_gradients[:, :, :, :] / float(self.n_samples)
        # if avg_gradients.max() != 0:
        #     avg_gradients /= avg_gradients.max()

        return avg_gradients
