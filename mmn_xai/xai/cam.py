""" Module to instantiate the CAM methods.

This module contains the methods to instantiate the CAM methods. The methods are stored in a
dictionary with the name of the method as key and the function to explain the image as value.
Within this category we have the following methods:
    - ScoreCAM
    - GradCAM
    - GradCAM++

Written by: Miquel Miró Nicolau (2023), UIB. From Rouen, France.
"""
import numpy as np
import pytorch_grad_cam as py_cam
import torch


def get_cam(img: np.ndarray, cam):
    """From a batch of images, get the saliency map for each image with a CAM method.

    Args:
        img: Numpy array or torch tensor with the images to explain.
        cam: Method to use to explain the image.

    Returns:
        Numpy array with the saliency map for each image.
    """
    result = []

    for i in range(img.shape[0]):
        explanation = cam(input_tensor=img[i: i + 1, :, :, :])
        result.append(explanation[0, :, :])
        torch.cuda.empty_cache()

    return result


def instantiate(
        net, device, layer, cuda_available, batch_size: int = 1, multi_channel: bool = False
):
    """Instantiate the CAM methods.

    Args:
        net: Pytorch model to explain.
        device: String with the GPU device to use.
        layer: Layer to use to explain the image.
        cuda_available: Boolean to indicate if the GPU is available.
        batch_size: Integer with the batch size to use.
        multi_channel: Boolean to indicate if the image has multiple channels.
    Returns:
        Dictionary with the CAM methods instantiated.
    """
    scam = py_cam.ScoreCAM(
        model=net,
        target_layers=layer,
        show_progress=False,
        device=device,
        batch_size=batch_size,
    )
    gcam = py_cam.GradCAM(
        model=net, target_layers=layer, use_cuda=cuda_available, device=device
    )
    gcam_plus = py_cam.GradCAMPlusPlus(
        model=net, target_layers=layer, use_cuda=cuda_available, device=device
    )

    acam = py_cam.AblationCAM(
        model=net, target_layers=layer, use_cuda=cuda_available, device=device
    )

    if not multi_channel:
        return {
            "score_cam": lambda x: get_cam(x[:, 0:1, :, :], scam),
            "grad_cam": lambda x: get_cam(x[:, 0:1, :, :], gcam),
            "grad_cam_plus": lambda x: get_cam(x.to(device), gcam_plus),
            "abblation_cam": lambda x: get_cam(x[:, 0:1, :, :], acam)
        }
    else:
        return {
            "score_cam": lambda x: get_cam(x[:, :, :, :], scam),
            "grad_cam": lambda x: get_cam(x[:, :, :, :], gcam),
            "grad_cam_plus": lambda x: get_cam(x.to(device), gcam_plus),
            "abblation_cam": lambda x: get_cam(x[:, :, :, :], acam)
        }
