""" Weighted Integrated Gradients.

Written by: Miquel Miró Nicolau (2023), UIB. From Rouen, France
"""
from typing import Callable

import torch

from mmn_xai.methods import gradient_methods


class WeightedIntegratedGrad(gradient_methods.GradientMethod):

    def __init__(
            self,
            pretrained_model,
            baseline: torch.Tensor = None,
            device: str = "cpu",
            add_softmax: bool = False,
            steps: int = 50,
            similarity_fn: Callable = None
    ):
        super().__init__(pretrained_model, device, add_softmax)
        self.baseline = baseline
        self.steps = steps

        if similarity_fn is None:
            similarity_fn = lambda x, y: 1 - min(abs((x - y).pow(2).sum().sqrt()), 1)

        self.similarity_fn = similarity_fn

    def __call__(self, image, *args, **kwargs):
        if self.baseline is None:
            self.baseline = 0 * image

        scaled_inputs = [self.baseline + (float(i) / self.steps) * (image - self.baseline) for i in
                         range(0, self.steps + 1)]
        explanation = torch.zeros(image)
        org_out = self.output_and_gradient(image)

        for inputs in scaled_inputs:
            output, grads = self.output_and_gradient(inputs)

            dist = self.similarity_fn(org_out, output)

            explanation += (dist * grads)

        explanation /= len(scaled_inputs)
        delta_x = image - self.baseline
        integrated_grad = delta_x * explanation

        return integrated_grad
