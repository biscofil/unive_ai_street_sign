from torch import Tensor


class FixTensorBrightnessContrast(object):

    def __call__(self, image: Tensor) -> Tensor:
        cp = image - image.min()
        cp = cp / cp.max()
        return cp
