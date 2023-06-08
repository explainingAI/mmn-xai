import numpy as np
import pytorch_grad_cam as py_cam
import torch


def get_cam(img: np.ndarray, cam):
    result = []

    for i in range(img.shape[0]):
        explanation = cam(input_tensor=img[i : i + 1, :, :, :])
        result.append(explanation[0, :, :])
        torch.cuda.empty_cache()

    return result


def instantiate(net, device, layer, cuda_available):
    scam = py_cam.ScoreCAM(
        model=net,
        target_layers=layer,
        use_cuda=cuda_available,
    )
    gcam = py_cam.GradCAM(model=net, target_layers=layer, use_cuda=cuda_available)
    gcam_plus = py_cam.GradCAMPlusPlus(
        model=net, target_layers=layer, use_cuda=cuda_available
    )

    return {
        "score_cam": lambda x: get_cam(x[:, 0:1, :, :], scam),
        "grad_cam": lambda x: get_cam(x[:, 0:1, :, :], gcam),
        "grad_cam_plus": lambda x: get_cam(x.to(device), gcam_plus),
    }
