""" Implementation of the SIDU method.

The SIDU methods is XAI method developed by Muddamsetty et al. (2021). The result is a saliency
map for a particular image.

Abstract:
    Explainable Artificial Intelligence (XAI) has in recent years become a well-suited framework to
    generate human understandable explanations of black box mod- els. In this paper, we present a
    novel XAI visual explanation algorithm denoted SIDU that can effectively localize entire object
    regions responsible for prediction in a full extend. We analyze its robustness and effectiveness
    through various computational and human subject experiments. In particular, we assess the SIDU
    algorithm using three different types of evaluations (Application, Human and
    Functionally-Grounded) to demonstrate its superior performance. The robustness of SIDU is
    further studied in presence of adversarial attack on black box models to better understand its
    performance.

References:
    Muddamsetty, S. M., Jahromi, M. N., Ciontos, A. E., Fenoy, L. M., & Moeslund, T. B. (2021).
    Introducing and assessing the explainable ai (mmn_xai) method: Sidu. arXiv preprint
    arXiv:2101.10710.

"""
from typing import Union

import numpy as np
import torch
from torch import nn


def generate_feature_activation_masks(
    conv_output: Union[np.array, torch.Tensor],
    image: torch.Tensor,
    device: torch.device,
    weights_thresh: Union[int, float, None] = 0.1,
):
    """
    Generate feature activation masks for an image.

    Args:
        conv_output: the output of the convolutional layer
        image: the input image
        weights_thresh: the threshold to apply to the convolutional output to create the mask
        device: Cuda device to use.

    Yields:
       A tuple of the feature activation mask and the image feature
    """
    # Apply the weight threshold to the convolutional output
    mask_w = conv_output
    mask_w = mask_w > weights_thresh
    mask_w = mask_w.type(torch.FloatTensor).to(device)

    # Resize the mask to the same size as the input image
    resize = nn.Upsample(tuple(image.shape[-2:]), mode="bilinear", align_corners=False)
    mask_w = resize(mask_w)

    # Batch, Filters, Channels, Width, Height
    for i in range(image.shape[1]):
        for j in range(mask_w.shape[1]):
            yield mask_w[0:1, j : j + 1, :, :], (
                image[0:1, :, :, :] * mask_w[0:1, j : j + 1, :, :]
            )


def sidu(
    model: torch.nn.Module,
    layer_output,
    image: Union[np.ndarray, torch.Tensor],
    device: torch.device = None,
):
    """
    Computes the SIDU feature map of an image given a pre-trained PyTorch model.

    Args:
        model (torch.nn.Module): A pre-trained PyTorch model used to compute the feature maps.
        layer_output: The layer output used to generate the feature maps.
        image (Union[np.ndarray, torch.Tensor]): A numpy array or PyTorch tensor representing the
                                                input image.
        device (torch.device, optional): The device to use for computations. If None, defaults to
                                        "cuda" if available, else "cpu".

    Returns:
        A PyTorch tensor representing the SIDU feature map of the input image.

    Raises:
        TypeError: If the `model` argument is not a PyTorch module or the `image` argument is not a
            numpy array or PyTorch tensor.
    Notes:
        The SIDU algorithm is used for generating visual explanations of deep neural networks. It
        combines information about the importance of image features (as measured by the SD score)
        with information about the similarity of predictions across different image feature
        activations (as measured by the U score), to generate a weighted sum of feature activation
        maps that captures the most semantically relevant features for a given input image.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image: torch.Tensor = image.to(device)
    predictions: list = []
    sd_score: list = []

    p_org: torch.Tensor = model(image).detach()
    for feature_activation_mask, image_feature in generate_feature_activation_masks(
        layer_output, image
    ):
        pred = model(image_feature).detach()
        sd_score.append(pred - p_org)

        predictions.append(pred)
    sd_score = torch.stack(sd_score).reshape((-1, 2))
    sd_score = torch.norm(sd_score, dim=1)
    sd_score = torch.exp((-1 / (2 * (0.25 ** 2))) * sd_score)

    u_score = torch.zeros((len(predictions), len(predictions))).to(device)
    predictions = torch.stack(predictions).to(device)
    for i in range(len(predictions)):
        u_score[i, :] = torch.norm(predictions[i] - predictions)

    u_score = torch.sum(u_score, axis=-1)
    weights = sd_score * u_score

    weighted_fams_tensor = torch.zeros_like(feature_activation_mask)

    for w, (fam, _) in zip(
        weights, generate_feature_activation_masks(layer_output, image, device=device)
    ):
        weighted_fams_tensor += (fam * w).detach()

    return weighted_fams_tensor


def sidu_wrapper(
    net: torch.nn.Module,
    layer,
    image: Union[np.array, torch.Tensor],
    device: torch.device = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    activation = {}

    def hook(model, input, output):
        activation["layer"] = output.detach()

    layer.register_forward_hook(hook)
    _ = net(image)

    return sidu(net, activation["layer"], image, device=device).cpu().detach().numpy()
