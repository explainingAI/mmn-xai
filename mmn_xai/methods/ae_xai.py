"""

Author: Miquel Miró
"""
import warnings

import torch
import numpy as np
from torch import nn
import torch.optim as optim

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier


class AEXAI:

    def __init__(
            self,
            window_size: int,
            model,
            input_shape,
            n_classes,
            device: torch.device = None,
    ):
        """

        Args:
            model (callable, optional): A function that takes a batch of images (shape (N, C, H, W)) and returns an
                OOD value per image.
            device (torch.device, optional): Device to use (e.g. torch.device("cuda")). If None, CUDA is used if
                available.
        """
        self.model = model
        self.art_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            input_shape=input_shape,
            nb_classes=n_classes,
        )

        self.window_size = window_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.__attack = FastGradientMethod(
            estimator=self.art_model,
            eps=1 / 255,
            minimal=True
        )

    def __call__(self, image):
        window_size = self.window_size

        max_img, min_img = image.max(), image.min()

        image = (image - min_img) / (max_img - min_img)
        image_perturbed = self.__attack.generate(x=image.detach().cpu())
        diff = abs(image - image_perturbed)

        if diff.max() > (1 / 255):
            warnings.warn(f"Perturbation higher than minimum (1/255): {diff.max()}")

        # Convert numpy array to torch.Tensor (if needed) and send to device.
        image = torch.from_numpy(image)
        image = image.to(self.device)
        image_perturbed = torch.from_numpy(image_perturbed)
        image_perturbed = image_perturbed.to(self.device)

        # Get image dimensions (assumed to be (C, H, W))
        C, H, W = image.shape
        # Number of sliding window positions
        H_out = H - window_size + 1
        W_out = W - window_size + 1

        original_output = self.model(image.reshape((1, C, H, W)).to(self.device)).detach()
        sal_map = torch.zeros_like(original_output)
        for i in range(H_out):
            for j in range(W_out):
                image_copy = np.copy(image)
                image_copy[:, i:i + window_size, j:j + window_size] = image_perturbed[i:i + window_size,
                                                                      j:j + window_size]

                output = self.model(torch.from_numpy(image_copy.reshape((1, C, H, W))).to(self.device)).detach()
                sal_map[i:i + window_size, j:j + window_size] += torch.mean(original_output - output)

        return sal_map
