""" Weighted SmoothGrad.

Ref: https://github.com/pkmr06/pytorch-smoothgrad/blob/master/lib/gradients.py
"""
import torch
from torch import nn
from torch.autograd import Variable

import numpy as np


class WeightedSmoothGrad:
    def __init__(
        self,
        pretrained_model,
        device="cpu",
        st_dev_spread=0.15,
        n_samples=25,
        magnitude=True,
        add_softmax: bool = False,
    ):
        self.pretrained_model = pretrained_model
        self.device = device
        self.st_dev_spread = st_dev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        self.add_softmax = add_softmax

    def gradient(self, image, index=None):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = Variable(image.to(self.device), requires_grad=True)
        output = self.pretrained_model(image)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(
            torch.from_numpy(one_hot).to(self.device), requires_grad=True
        )
        one_hot = torch.sum(one_hot * output)

        if image.grad is not None:
            image.grad.data.zero_()
        one_hot.backward(retain_graph=True)

        grad = image.grad.data.cpu().numpy()

        return output, grad

    def __call__(self, x, index=None):
        output, grad = self.gradient(x.to(self.device))
        org_output = output

        if self.add_softmax:
            org_output = nn.functional.softmax(org_output)

        x = x.data.cpu().numpy()
        stdev = self.st_dev_spread * (np.max(x) - np.min(x))

        total_gradients = np.zeros_like(x)
        for _ in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            output, grad = self.gradient(x_plus_noise)
            if self.add_softmax:
                output = nn.functional.softmax(output)

            dist = 1 - min(abs((org_output - output).pow(2).sum().sqrt()), 1)
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
