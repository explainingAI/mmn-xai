""" Module containing the methods to explain the images.

This module contains the methods to explain the images. The methods are stored in a dictionary
with the name of the method as key and the function to explain the image as value.

Written by: Miquel Miró-Nicolau (2023), UIB.
"""
from captum import attr

from mmn_xai.methods import utils
from mmn_xai.methods import w_smooth_grad as wsg


def instantiate(net, device):
    net = net.to(device)

    lrp = attr.LRP(net)
    ig = attr.IntegratedGradients(net)
    sal = attr.Saliency(net)
    gbp = attr.GuidedBackprop(net)
    deep_lift = attr.DeepLift(net)
    smooth_grad = attr.NoiseTunnel(sal)
    wsg_inst = wsg.WeightedSmoothGrad(net, device=device)

    methods = {
        "lrp": lambda x: abs(
            utils.to_numpy(lrp.attribute(x.to(device), target=0)[:, 0, :, :])
        ),
        "integrated_gradients": lambda x: abs(
            utils.to_numpy(ig.attribute(x.to(device), target=0)[:, 0, :, :])
        ),
        "gradient": lambda x: abs(
            utils.to_numpy(sal.attribute(x.to(device), target=0)[:, 0, :, :])
        ),
        "gbp": lambda x: abs(utils.to_numpy(gbp.attribute(x.to(device), target=0)[:, 0, :, :])),
        "DeepLift": lambda x: abs(
            utils.to_numpy(deep_lift.attribute(x.to(device), target=0)[:, 0, :, :])
        ),
        "SmoothGrad": lambda x: abs(
            utils.to_numpy(
                smooth_grad.attribute(
                    x.to(device), nt_type="smoothgrad", nt_samples=25, target=0
                )[:, 0, :, :]
            )
        ),
        "WeightedSmoothGrad": lambda x: abs(wsg_inst(x)),
    }

    return methods
