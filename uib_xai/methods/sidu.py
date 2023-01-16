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
    Introducing and assessing the explainable ai (uib_xai) method: Sidu. arXiv preprint
    arXiv:2101.10710.

"""
from typing import Union

import numpy as np
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_feature_activations_masks(
        conv_output: Union[np.array, torch.Tensor],
        image: torch.Tensor,
        weights_thresh: Union[int, float, None] = 0.1,
):
    """

    Args:
        conv_output:
        image:
        weights_thresh:

    Returns:

    """
    mask_w = conv_output

    mask_w = mask_w > weights_thresh
    mask_w = mask_w.type(torch.FloatTensor)
    resize = nn.Upsample(tuple(image.shape[-2:]), mode="bilinear", align_corners=False)
    mask_w = resize(mask_w)

    feature_activation_masks = mask_w

    # Batch, Filters, Channels, Width, Height
    image = image.reshape(
        (image.shape[0], 1, image.shape[1], image.shape[2], image.shape[3])
    )
    mask_w = mask_w.reshape(
        (mask_w.shape[0], mask_w.shape[1], 1, mask_w.shape[2], mask_w.shape[3])
    )
    image = image.repeat((1, mask_w.shape[1], 1, 1, 1))
    mask_w = mask_w.repeat((1, 1, image.shape[2], 1, 1))

    mask_w = mask_w.to(DEVICE)

    image_features = image * mask_w

    return feature_activation_masks, image_features


def similarity_difference(model, org_img, feature_activation_masks, sigma):
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
        pred_diffs.append(prediction - p_org)

    pred_diffs = torch.stack(pred_diffs)
    similarity_diff = torch.norm(pred_diffs, dim=2)
    similarity_diff = torch.exp((-1 / (2 * (sigma ** 2))) * similarity_diff)

    return similarity_diff.to(DEVICE)


def uniqueness(model, feature_activation_masks):
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

    return torch.Tensor(uniqueness_score).to(DEVICE)


def sidu(model: torch.nn.Module, layer_output, image: Union[np.ndarray, torch.Tensor]):
    """ SIDU method.

    This method is an XAI method developed by Muddamsetty et al. (2021). The result is a saliency
    map for a particular image.

    Args:
        model:
        layer_output:
        image:

    Returns:

    """
    feature_activation_masks, image_features = get_feature_activations_masks(
        layer_output, image
    )

    sd_score = similarity_difference(model, image, image_features, sigma=0.25)
    u_score = uniqueness(model, image_features)

    weights = [(sd_i * u_i) for sd_i, u_i in zip(sd_score, u_score)]
    weighted_fams = [
        fam * w for fam, w in zip(torch.squeeze(feature_activation_masks), weights)
    ]
    weighted_fams_tensor = torch.stack(weighted_fams)

    explanation = torch.sum(weighted_fams_tensor, axis=0)

    return explanation


def sidu_wrapper(net: torch.nn.Module, layer, image: Union[np.array, torch.Tensor]):
    activation = {}

    def hook(model, input, output):
        activation["layer"] = output.detach()

    layer.register_forward_hook(hook)

    return sidu(net, activation["layer"], image).cpu().detach().numpy()
