# -*- coding: utf-8 -*-
""" Module containing a set of XAI algorithms.

Explainable Artificial Intelligence (XAI) is a collection of algorithm aiming to open black-box
models (as neural networks) and expose them to be explainable. In this module we provide a set
of methods:
    GradCAM
    LIME
    Occlusion test

Writen by: Miquel Miró Nicolau (UIB)
"""
import numpy as np
import tensorflow as tf


def grad_cam(img_array: np.ndarray, model, last_conv_layer_name, pred_index=None) -> np.ndarray:
    """ GradCAM implementation.

    Gradient-weighted Class Activation Mapping (Grad-CAM), uses the class-specific gradient
    information flowing into the final convolutional layer of a CNN to produce a coarse localization
    map of the important regions in the image. Grad-CAM is a strict generalization of the Class
    Activation Mapping. Unlike CAM, Grad-CAM requires no re-training and is broadly applicable to
    any CNN-based architectures.

    Proposed by:
        F. Chollet. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
            Localization.
    Args:
        img_array:
        model:
        last_conv_layer_name:
        pred_index:

    Returns:
        heatmap
    """
    # First, we create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image with respect to
    # the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen) with regard to the output
    # feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific
    # feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array by "how important this channel is" with
    # regard to the top predicted class then sum all the channels to obtain the heatmap class
    # activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def build_grid(stride, size, image_shape):
    height, width, _ = image_shape

    start_cx = np.arange(0, width, size * stride)
    start_cy = np.arange(0, height, size * stride)
    start_cx, start_cy = np.meshgrid(start_cx, start_cy)

    box_widths, box_center_x = np.meshgrid(np.array([size]), start_cx)
    box_heights, box_center_y = np.meshgrid(np.array([size]), start_cy)

    box_start = np.stack([box_center_x, box_center_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    box_end = box_start + box_sizes
    boxes = np.concatenate([box_start, box_end], axis=1)

    return boxes


def print_bboxes(bboxes, shape):
    """ Draw bounding boxes on an image.

    Args:
        bboxes: List of bounding boxes in the form [x1, y1, x2, y2].
        shape: Tuple containing the image shape.

    Returns:
        Image with bounding boxes drawn on it.
    """
    bboxes_img = np.zeros(shape)
    for idx, bb in enumerate(bboxes):
        bboxes_img[bb[0]: bb[2], bb[1]:bb[3]] = idx + 1

    return bboxes_img


def occlusion_test(batch_img, model, occlusion_zones, intermediate_value=None):
    """ Occlusion test for a batch of images.

    The occlusion test is a simple way to test the robustness of the model to occlusion. Occlusion
    in the context of computer vision is the loss of information in an image by removing a part, in
    this case we substitute the part of the image with a box with the intermediate value passed as
    parameter.

    Args:
        batch_img: NumPy array with the batch of images to test.
        model: Model, with a predict method to be tested against the occlusion.
        occlusion_zones:
        intermediate_value:

    Returns:

    """
    image_shape = batch_img.shape

    batch_img = np.copy(batch_img.reshape((1, image_shape[1], image_shape[2], image_shape[3])))
    original_prop = model.predict(batch_img)[0]

    # Getting the index of the winning class:
    index_object = np.argmax(original_prop)
    _, height, width, _ = batch_img.shape

    heatmap = np.zeros((batch_img.shape[1], batch_img.shape[2], batch_img.shape[3]),
                       dtype=np.float64)

    if intermediate_value is None:
        intermediate_value = (batch_img.max() - batch_img.min()) / 2

    for u_val in np.unique(occlusion_zones):
        mask = occlusion_zones == u_val

        img_ocluded = np.copy(batch_img)

        img_ocluded[0, :, :, 0][mask] = intermediate_value
        img_ocluded[0, :, :, 1][mask] = intermediate_value
        img_ocluded[0, :, :, 2][mask] = intermediate_value

        oclussion_prop = model.predict(img_ocluded)[0]
        oclussion_prop = (original_prop[index_object] - oclussion_prop[index_object])

        heatmap[:, :, 0][mask] = oclussion_prop
        heatmap[:, :, 1][mask] = oclussion_prop
        heatmap[:, :, 2][mask] = oclussion_prop

    heatmap /= heatmap.max()
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)

    return heatmap
