import abc

import numpy as np
import torch
from torch.autograd import Variable


class GradientMethod(abc.ABC):
    def __init__(
        self, pretrained_model: torch.nn.Module, device="cpu", add_softmax: bool = False
    ):
        self.pretrained_model = pretrained_model
        self.device = device
        self.add_softmax = add_softmax

    def output_and_gradient(self, image, index=None):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = Variable(image.to(self.device), requires_grad=True)
        output = self.pretrained_model(image)

        if index is None:
            index = torch.argmax(output.data, axis=1)

        one_hot = torch.zeros(
            (output.size()[0], output.size()[-1]), dtype=torch.float32
        )
        for batch_idx, inter_index in enumerate(index):
            one_hot[batch_idx][inter_index] = 1

        one_hot = Variable(one_hot.to(self.device), requires_grad=True)
        # Només ens quedam amb la sortida amb més gradient
        one_hot = torch.sum(one_hot * output)

        if image.grad is not None:
            image.grad.data.zero_()
        one_hot.backward(retain_graph=True)

        grad = image.grad.data

        return output, grad

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplemented
