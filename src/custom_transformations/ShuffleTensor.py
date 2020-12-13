import torch
from torch import Tensor


class ShuffleTensor:

    def __call__(self, t: Tensor):
        return t[torch.randperm(t.size()[0])]
