# -*- coding: utf-8 -*-
""" Module containing a set of XAI algorithms.

Explainable Artificial Intelligence (XAI) is a collection of algorithm aiming to open black-box
models (as neural networks) and expose them to be explainable. In this module we provide a set
of methods:
    GradCAM
    Grad-CAM++
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

    Implemented by:
        F. Chollet. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
            Localization.
    Args:
        img_array: Numpy array containing the image.
        model: Keras model.
        last_conv_layer_name: String containing the name of the last convolutional layer.
        pred_index: Index of the class to be used for the prediction.

    Returns:
        Numpy array containing the image with the heatmap.
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
        Numpy array containing the image with the heatmap.
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


def grad_cam_plus(model, img, layer_name, label_name=None, category_id=None):
    """ Get a heatmap by Grad-CAM++.

    Grad-CAM++ that can provide better visual explanations of CNN model predictions, in terms of
    better object localization as well as explaining occurrences of multiple object instances in a
    single image, when compared to state-of-the-art.

    Refs:
        https://arxiv.org/abs/1710.11063

    Implemented by:
        Samson Woof, https://github.com/samson6460/tf_keras_gradcamplusplus

    Args:
        model: A model object, build from tf.keras 2.X.
        img: Numpy array with the image to be processed.
        layer_name: String with the name of the layer to be used.
        label_name: A list or None, show the label name by assign this argument, it should be a list
                    of all label names.
        category_id: An integer, index of the class. Default is the category with the highest score
                    in the prediction.
    Return:
        Numpy array containing the image with the heatmap.
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gradient_tape1:
        with tf.GradientTape() as gradient_tape2:
            with tf.GradientTape() as gradient_tape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                if label_name:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gradient_tape3.gradient(output, conv_output)
            conv_second_grad = gradient_tape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gradient_tape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denominator = conv_second_grad[0] * 2.0 + conv_third_grad[0] * global_sum
    alpha_denominator = np.where(alpha_denominator != 0.0, alpha_denominator, 1e-10)

    alphas = alpha_num / alpha_denominator
    alpha_normalization_constant = np.sum(alphas, axis=(0, 1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights * alphas, axis=(0, 1))
    grad_cam_map = np.sum(deep_linearization_weights * conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap
