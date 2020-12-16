import itertools

import torch
from torch import Tensor, nn


class SlidingWindow(object):

    def __init__(self, model: nn.Module, device_name: str, patch_size: int, patch_stride: int):
        super().__init__()
        self.device = device_name
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.model = model

    @staticmethod
    def get_sizes(original_w: int, original_h: int,
                  patch_size: int, patch_stride: int,
                  min_w: int = 100, min_h: int = 100, n_res=7) -> list:
        max_h = original_h / 3

        # delta_w = int((original_w - min_w) / n_res)
        # w_range = list(range(min_w, int(original_w), delta_w))

        rng = [min_h + (max_h - min_h) / n_res * r for r in range(0, n_res, 1)]

        ratio_range = [h / original_h for h in rng]

        sizes = [(int(original_w * r), int(original_h * r), r) for r in ratio_range]
        sizes = [(x, y, r, SlidingWindow.get_grid(x, y, patch_size, patch_stride)) for x, y, r in sizes]
        return sizes

    @staticmethod
    def get_grid(img_width: int, img_height: int, patch_size: int, patch_stride: int) -> list:
        # [x1,x2,...,xn] m times
        x_ = list(range(0, img_width + 1 - patch_size, patch_stride))
        # [y1 (n times),y2 (n times), ..., ym (n times)]
        y_ = list(range(0, img_height + 1 - patch_size, patch_stride))
        m, n = len(y_), len(x_)
        x_ = x_ * m
        y_ = [[v] * n for v in y_]
        y_ = list(itertools.chain(*y_))
        return [(x, y) for x, y in zip(x_, y_)]

    def unfold_and_classify(self, x: Tensor) -> (list, list):
        x.detach()
        c, h, w = x.shape

        x = x.detach()
        patches = x.permute(1, 2, 0) \
            .unfold(0, self.patch_size, self.patch_stride) \
            .unfold(1, self.patch_size, self.patch_stride) \
            .reshape(-1, c, self.patch_size, self.patch_size)

        # split into batches to prevent filling up the whole GPU memory
        patch_batches = torch.split(patches, 400, dim=1)
        labels = []
        scores = []
        for patch_batch in patch_batches:
            l, s = self.model.get_tensor_label(patch_batch)
            labels += l
            scores += s
        return labels, scores
