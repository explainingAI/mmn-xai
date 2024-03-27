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

from captum import attr
from lime.lime_image import LimeImageExplainer
import numpy as np
import torch

from mmn_xai.methods import lime, rise, sidu
from mmn_xai.methods import utils


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

def zeiler_fn(net, data, target, sliding_window_shapes, device):
    zeiler = attr.Occlusion(net)
    expl = __occlusion_expl(
        lambda y: zeiler.attribute(y, target=0, sliding_window_shapes=sliding_window_shapes)[:, 0, :, :], data, device
    )
    
    del zeiler
    
    return expl
    

def instantiate(net, device, layer, rise_size: tuple, rise_n: int,  multi_channel: bool, mask_path: str = "masks.npy", batch_size: int = None):
    """ Instantiate the occlusion methods.

    Args:
        net:
        device:
        layer:
        mask_path:

    Returns:

    """
    explainer_rise = rise.RISE(net, rise_size, gpu_batch=1, device=device).to(device)
    explainer_rise.generate_masks(N=rise_n, s=8, p1=0.1, savepath=mask_path)
    # zeiler = attr.Occlusion(net)
    
    channel =  1
    if multi_channel:
        channel = 3

    return {
        # "zeiler": lambda x: utils.to_numpy(zeiler.attribute(x.to(device), target=0, sliding_window_shapes=(channel,64,64), show_progress = True)[:, 0, :, :]),
        "zeiler": lambda x: zeiler_fn(net, x.to(device), target=0, sliding_window_shapes=(channel, 64, 64), device=device)[0],
        "rise": lambda x: __occlusion_expl(
            lambda y: explainer_rise(y)[0, :, :], x, device
        ),
        "lime": lambda x: __occlusion_expl(
            lambda y: lime.get_exp(
                explainer=LimeImageExplainer(),
                img=y[0, 0, :, :],
                # img = y,
                net=net,
                device=device,
                hide_color_fn=1,
                segmentation_fn=None,
                num_samples=1000,
                multi_channel= multi_channel,
                batch_size=batch_size,
            )[0],
            x,
            device,
        ),
        "sidu": lambda x: sidu.sidu_wrapper(net, layer, x.to(device), device)[0],
    }
