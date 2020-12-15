import torch
from torch import Tensor, nn


class CnnDetectorWithSlidingWindow(object):

    def __init__(self, model: nn.Module, device_name: str, patch_size: int, patch_stride: int):
        super().__init__()
        self.device = device_name
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.model = model

    def unfold_and_classify(self, x: Tensor) -> (list, list):
        x.detach()
        c, h, w = x.shape

        x = x.detach()
        patches = x.permute(1, 2, 0) \
            .unfold(0, self.patch_size, self.patch_stride) \
            .unfold(1, self.patch_size, self.patch_stride) \
            .reshape(-1, c, self.patch_size, self.patch_size)

        # split into batches
        patch_batches = torch.split(patches, 400, dim=1)
        labels = []
        scores = []
        for patch_batch in patch_batches:
            l, s = self.model.get_tensor_label(patch_batch)  # TODO check +200MB
            labels += l
            scores += s
        return labels, scores
