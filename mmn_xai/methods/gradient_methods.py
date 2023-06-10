import abc

import numpy as np
import torch
from torch.autograd import Variable

class GradientMethod(abc.ABC):
    def __init__(
            self,
            pretrained_model: torch.Module,
            device="cpu",
            add_softmax: bool = False
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
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(
            torch.from_numpy(one_hot).to(self.device), requires_grad=True
        )
        one_hot = torch.sum(one_hot * output)

        if image.grad is not None:
            image.grad.data.zero_()
        one_hot.backward(retain_graph=True)

        grad = image.grad.data.cpu().numpy()

        return output, grad

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplemented
