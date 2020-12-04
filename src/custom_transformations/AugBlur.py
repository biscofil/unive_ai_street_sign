from PIL import ImageFilter
from PIL.Image import Image


class AugBlur(object):

    def __call__(self, image: Image) -> Image:
        return image.filter(ImageFilter.BLUR)
