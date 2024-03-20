""" Module to instantiate the occlusion methods.

This module contains the methods to instantiate the occlusion methods. The methods are stored in a
dictionary with the name of the method as key and the function to explain the image as value.
Within this category we have the following methods:
    - RISE
    - SIDU
    - LIME

Written by: Miquel Miró Nicolau (2023), UIB. From Rouen, France.
"""
from typing import Union

import numpy as np
import torch
from captum import attr
from lime.lime_image import LimeImageExplainer

from mmn_xai.methods import lime, rise, sidu


def zeiler_fn(net, data, sliding_window_shapes, device) -> np.ndarray:
    """ Function to explain the image with the Zeiler et al. method.

    Args:
        net: Pytorch model to explain.
        data: Image to explain.
        sliding_window_shapes: Tuple with the sliding window shapes for the Zeiler et al. method.
        device: String with the GPU device to use.

    Returns:
        Numpy array with the saliency map for the image passed as parameter.
    """
    zeiler = attr.Occlusion(net)
    expl = __occlusion_expl(
        lambda y: zeiler.attribute(
            y, target=0, sliding_window_shapes=sliding_window_shapes
        )[:, 0, :, :],
        data,
        device,
    )
    return expl


def __occlusion_expl(
    explainer, batch: Union[np.ndarray, torch.Tensor], device: str
) -> np.ndarray:
    """Standardizes RISE output.

    Args:
        rise_exec: Instance of explainer.
        batch (np.array or torch.Tensor): Image to explain.
        device (str): String to define the device to use in PyTorch format.

    Returns:
        Numpy array with the saliency map for the image passed as parameter.
    """

    results = []

    with torch.no_grad():
        if not isinstance(batch, torch.Tensor):
            batch = torch.Tensor(batch)
        batch = batch.to(device)

        for i in range(batch.shape[0]):
            image = batch.type(torch.float32)[i : i + 1, :, :, :]
            explanation = explainer(image)

            if isinstance(explanation, torch.Tensor):
                explanation = explanation.detach().cpu().numpy()

            results.append(explanation)
            torch.cuda.empty_cache()

    return abs(np.asarray(results))


def instantiate(
    net,
    device,
    layer,
    batch_size: int,
    rise_size: tuple = (128, 128),
    rise_n: int = 6000,
    mask_path: str = "masks.npy",
    sliding_window_shapes: tuple = (1, 64, 64),
):
    """Instantiate the occlusion methods.

    Args:
        net: Pytorch model to explain.
        device: String with the GPU device to use.
        layer: Layer to use to explain the image.
        rise_size: Tuple with the size of the RISE mask.
        rise_n: Integer with the number of RISE masks to generate.
        mask_path: String with the path to save the RISE masks.
        sliding_window_shapes: Tuple with the sliding window shapes for the Zeiler et al. method.
        batch_size: Integer with the batch size to use.

    Returns:
        Dictionary with the occlusion methods instantiated.
    """
    explainer_rise = rise.RISE(net, rise_size, gpu_batch=batch_size, device=device).to(
        device
    )
    explainer_rise.generate_masks(N=rise_n, s=8, p1=0.1, savepath=mask_path)

    return {
        "zeiler": lambda x: zeiler_fn(
            net,
            x.to(device),
            sliding_window_shapes=sliding_window_shapes,
            device=device,
        )[0],
        "rise": lambda x: __occlusion_expl(
            lambda y: explainer_rise(y)[0, :, :], x, device
        ),
        "lime": lambda x: __occlusion_expl(
            lambda y: lime.get_exp(
                explainer=LimeImageExplainer(),
                img=y[0, 0, :, :],
                net=net,
                device=device,
                hide_color_fn=1,
                segmentation_fn=None,
                num_samples=1000,
                batch_size=batch_size,
            )[0],
            x,
            device,
        ),
        "sidu": lambda x: sidu.sidu_wrapper(net, layer, x.to(device), device)[0],
    }
