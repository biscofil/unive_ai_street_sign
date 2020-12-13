from PIL import ImageOps
from PIL.Image import Image


class AugFlip(object):

    def __call__(self, image: Image) -> Image:
        return ImageOps.flip(image)
