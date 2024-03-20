""" Module containing the methods to explain the images.

This module contains the methods to explain the images. The methods are stored in a dictionary
with the name of the method as key and the function to explain the image as value.

Written by: Miquel Miró-Nicolau (2023), UIB.
"""
import torch

from mmn_xai.xai import gradients, occlusion, cam


def instantiate(
    net,
    layer=None,
    device="cpu",
    sidu_layer=None,
    do_cam: bool = True,
    do_grad: bool = True,
    do_occlusion: bool = True,
    multi_channel: bool = False,
    rise_size: tuple = (128, 128),
    rise_n: int = 6000,
    batch_size: int = None,
):
    """instantiate all models.

    Args:
        net: Pytorch model
        layer: For CAM xai, indicate the layer to explain
        device: Pytorch device, if not set use the CPU
        sidu_layer: Layer for SIDU method
        do_cam: Flag, if true instantiate CAM xai.
        do_grad: Flag, if true instantiate gradient xai.
        do_occlusion: Flag, if true instantiate occlusion xai.

    Returns:
        Dictionary name-> method
    """
    net = net.to(device)

    if layer is None:
        layer = [net.module.maxpool2] if hasattr(net, "module") else [net.maxpool2]

    methods = {}

    if do_cam:
        cam_methods = cam.instantiate(net, device, layer, torch.cuda.is_available(), multi_channel)
        methods = {**methods, **cam_methods}

    if do_grad:
        grad_methods = gradients.instantiate(net, device)
        methods = {**methods, **grad_methods}

    if do_occlusion:
        occl_methods = occlusion.instantiate(
            net,
            device,
            sidu_layer,
            rise_size=rise_size,
            rise_n=rise_n,
            batch_size=batch_size,
        )
        methods = {**methods, **occl_methods}

    return methods
