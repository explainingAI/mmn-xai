# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def grad_cam(img_array: np.ndarray, model, last_conv_layer_name, pred_index=None) -> np.ndarray:
    """

    Gradient-weighted Class Activation Mapping (Grad-CAM), uses the class-specific gradient
    information flowing into the final convolutional layer of a CNN to produce a coarse localization
    map of the important regions in the image. Grad-CAM is a strict generalization of the Class
    Activation Mapping. Unlike CAM, Grad-CAM requires no re-training and is broadly applicable to
    any CNN-based architectures.

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


def occlusion_test(img, model, stride=1, size=3):
    original_prop = model(img)[0]
    # Getting the index of the winning class:
    index_object = np.argmax(original_prop)
    height, width, _ = img.shape

    heatmap = np.zeros_like(img)

    start_cx = np.arange(0, width, size * stride)
    start_cy = np.arange(0, height, size * stride)
    start_cx, start_cy = np.meshgrid(start_cx, start_cy)

    box_widths, box_center_x = np.meshgrid(np.array([size]), start_cx)
    box_heights, box_center_y = np.meshgrid(np.array([size]), start_cy)

    box_start = np.stack([box_center_x, box_center_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    box_end = box_start + box_sizes
    boxes = np.concatenate([box_start, box_end], axis=1)

    for box in boxes:
        img_ocluded = np.copy(img)
        img_ocluded[box[0]:box[2], box[1]:box[3], :] = 0

        oclussion_prop = model(img_ocluded)[0]
        oclussion_prop = (original_prop[index_object] - oclussion_prop[index_object]) / \
                         original_prop[index_object]

        heatmap[box[0]:box[2], box[1]:box[3], :] = oclussion_prop

    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)

    return heatmap
