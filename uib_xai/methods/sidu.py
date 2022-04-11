# -*- coding: utf-8 -*-
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
import cv2

import torch


def get_feature_activations_masks(conv_output, image: torch.Tensor,
                                  weights_thresh: Union[int, float, None] = None):
    """

    Args:
        conv_output:
        image:
        weights_thresh:

    Returns:

    """
    conv_output_np = conv_output.cpu().detach().numpy()
    conv_output_np = conv_output_np.reshape(-1, conv_output_np.shape[-2],
                                                conv_output_np.shape[-1])

    feature_activation_masks = []
    image_features = []
    for i in range(conv_output_np.shape[0]):
        mask_w = conv_output_np[i, :, :]

        if weights_thresh is None:
            weights_thresh = np.quantile(mask_w, 0.5)

        mask_w = mask_w > weights_thresh
        mask_w = mask_w.astype(np.float32)
        mask_w = cv2.resize(mask_w, image.cpu().numpy().shape[-2:], interpolation=cv2.INTER_LINEAR)

        feature_activation_masks.append(torch.tensor(mask_w))
        if len(image.shape) > 2:
            img_feat = image.cpu().numpy() * np.repeat(mask_w[np.newaxis, :, :], 3, axis=0)
        else:
            img_feat = image.cpu().numpy() * mask_w

        image_features.append(torch.tensor(img_feat))

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

    predictions = [model(fam) for fam in feature_activation_masks]
    pred_diffs = [(p_org - pi) for pi in predictions]

    similarity_diff = np.array([np.linalg.norm(pi.cpu().detach().numpy()) for pi in pred_diffs])
    similarity_diff = np.exp((-1 / (2 * (sigma ** 2))) * similarity_diff)

    return similarity_diff


def uniqueness(model, feature_activation_masks):
    """ Calculate the uniqueness of the feature activation masks.

    Args:
        model:
        feature_activation_masks:

    Returns:

    """
    predictions = [model(fam).cpu().detach().numpy() for fam in feature_activation_masks]
    uniqueness_score = []

    for i in range(len(predictions)):
        i_uniq = 0
        for j in range(len(predictions)):
            if i != j:
                i_uniq += np.linalg.norm(predictions[i] - predictions[j])
        uniqueness_score.append(i_uniq)

    return uniqueness_score


def sidu(model, layer_output, image: Union[np.ndarray, torch.Tensor]):
    feature_activation_masks, image_features = get_feature_activations_masks(layer_output, image)

    sd_score = similarity_difference(model, image, image_features, sigma=0.25)
    u_score = uniqueness(model, image_features)

    weights = np.array([np.dot(sd_i, u_i) for sd_i, u_i in zip(sd_score, u_score)])
    weighted_fams = [fam * w for fam, w in zip(feature_activation_masks, weights)]

    explanation = np.sum(weighted_fams, axis=0)

    return explanation
