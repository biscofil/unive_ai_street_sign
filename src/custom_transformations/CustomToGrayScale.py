from PIL import ImageOps
from PIL.Image import Image


class CustomToGrayScale(object):

    def __call__(self, pil_image: Image) -> Image:
        return ImageOps.grayscale(pil_image)
