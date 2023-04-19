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

    Returns:

    """
    # Apply the weight threshold to the convolutional output
    feature_activation_masks = conv_output
    feature_activation_masks = feature_activation_masks > weights_thresh
    feature_activation_masks = feature_activation_masks.type(torch.FloatTensor).to(
        device
    )

    # Resize the mask to the same size as the input image
    resize = nn.Upsample(tuple(image.shape[-2:]), mode="bilinear", align_corners=False)
    feature_activation_masks = resize(feature_activation_masks)

    # Batch, Filters, Channels, Width, Height
    image = image.reshape(
        (image.shape[0], 1, image.shape[1], image.shape[2], image.shape[3])
    )
    feature_activation_masks = feature_activation_masks.reshape(
        (
            feature_activation_masks.shape[0],
            feature_activation_masks.shape[1],
            1,
            feature_activation_masks.shape[2],
            feature_activation_masks.shape[3],
        )
    )
    image = image.repeat((1, feature_activation_masks.shape[1], 1, 1, 1))
    feature_activation_masks = feature_activation_masks.repeat(
        (1, 1, image.shape[2], 1, 1)
    )

    feature_activation_masks = feature_activation_masks.to(device)

    image_features = image * feature_activation_masks

    return feature_activation_masks, image_features


def similarity_difference(
    model, org_img, feature_activation_masks, sigma, device: torch.device
):
    """

    Args:
        model:
        org_img:
        feature_activation_masks:
        sigma:

    Returns:

    """
    p_org = model(org_img)
    pred_diffs = []
    for fam in torch.squeeze(feature_activation_masks):
        prediction = model(torch.unsqueeze(fam, 0))
        pred_diffs.append(p_org - prediction)

    pred_diffs = torch.stack(pred_diffs)
    similarity_diff = torch.norm(pred_diffs, dim=2)
    similarity_diff = torch.exp((-1 / (2 * (sigma ** 2))) * similarity_diff)

    return similarity_diff.to(device)


def uniqueness(
    model,
    feature_activation_masks,
    device: torch.device,
):
    """Calculate the uniqueness of the feature activation masks.

    Args:
        model:
        feature_activation_masks:

    Returns:

    """
    predictions = [
        model(torch.unsqueeze(fam, 0))
        for fam in torch.squeeze(feature_activation_masks)
    ]
    uniqueness_score = []

    for i in range(len(predictions)):
        i_uniq = 0
        for j in range(len(predictions)):
            if i != j:
                i_uniq += torch.norm(predictions[i] - predictions[j])
        uniqueness_score.append(i_uniq)

    return torch.Tensor(uniqueness_score).to(device)


def sidu(
    model: torch.nn.Module,
    layer_output,
    image: Union[np.ndarray, torch.Tensor],
    device: torch.device,
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
    feature_activation_masks, image_features = generate_feature_activation_masks(
        layer_output, image, device
    )

    sd_score = similarity_difference(
        model, image, image_features, sigma=0.25, device=device
    )
    u_score = uniqueness(model, image_features, device)

    weights = [(sd_i * u_i) for sd_i, u_i in zip(sd_score, u_score)]
    weighted_fams = [
        fam * w for fam, w in zip(torch.squeeze(feature_activation_masks), weights)
    ]
    weighted_fams_tensor = torch.stack(weighted_fams)

    explanation = torch.sum(weighted_fams_tensor, axis=0)

    return explanation


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

    return sidu(net, activation["layer"], image, device).cpu().detach().numpy()
