from torch import Tensor
from torch.nn import Module


class DebugLayer(Module):

    def __init__(self, info: str):
        super().__init__()
        self.info = info

    def forward(self, t: Tensor) -> Tensor:
        print("\t >>>", self.info, t.shape)
        return t
