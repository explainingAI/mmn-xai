from typing import Union

from lime.lime_image import LimeImageExplainer
import numpy as np
import torch

from mmn_xai.methods import lime, rise, sidu


def __occlusion_expl(
    explainer, batch: Union[np.ndarray, torch.Tensor], device: str
) -> np.ndarray:
    """Standardizes RISE output.

    Args:
        rise_exec: Instance of explainer.
        batch (np.array or torch.Tensor): Image to explain.
        device (str): String to define the device to use in PyTorch format.

    Returns:
        Numpy array with the saliency map for the image passed as parameter.
    """

    results = []

    with torch.no_grad():
        if not isinstance(batch, torch.Tensor):
            batch = torch.Tensor(batch)
        batch = batch.to(device)

        for i in range(batch.shape[0]):
            image = batch.type(torch.float32)[i : i + 1, :, :, :]
            explanation = explainer(image)

            if isinstance(explanation, torch.Tensor):
                explanation = explanation.detach().cpu().numpy()

            results.append(explanation)
            torch.cuda.empty_cache()

    return abs(np.asarray(results))


def instantiate(net, device, layer, mask_path: str = "masks.npy"):
    explainer_rise = rise.RISE(net, (128, 128), device=device).to(device)
    explainer_rise.generate_masks(N=6000, s=8, p1=0.1, savepath=mask_path)

    return {
        "rise": lambda x: __occlusion_expl(
            lambda y: explainer_rise(y)[0, :, :], x, device
        ),
        "lime": lambda x: __occlusion_expl(
            lambda y: lime.get_exp(
                explainer=LimeImageExplainer(),
                img=y[0, 0, :, :],
                net=net,
                device=device,
                hide_color_fn=1,
                segmentation_fn=None,
                num_samples=1000,
            )[0],
            x,
            device,
        ),
        "sidu": lambda x: sidu.sidu_wrapper(net, layer, x.to(device), device)[0],
    }
