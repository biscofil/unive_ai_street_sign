import numpy as np
import torch
from torch import Tensor


class CustomToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, pil_image) -> Tensor:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        np_image = np.array(pil_image)

        image_transposed = np_image.transpose((2, 0, 1))

        img_float = image_transposed.astype(np.float) / 255.0

        return torch.from_numpy(img_float).float()
