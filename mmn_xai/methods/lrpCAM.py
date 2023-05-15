from typing import Union

import torch
from captum import attr

from . import utils


class lrpCAM:

    def __init__(self, model, sub_model):
        self.__model = model
        self.__sub_model = sub_model

    def __call__(self, image: torch.Tensor, target: Union[int, None], layer, *args, **kwargs):
        lrp = attr.LRP(self.__sub_model)

        activation = utils.get_activation(image, layer, model=self.__model)
        relevance = lrp.attribute(torch.flatten(activation), target=target, *args, **kwargs)

        unflat = torch.nn.Unflatten(0, activation.shape)
        relevance = unflat(relevance)

        return torch.sum(relevance * activation, dim=1)[0]
