import random

from PIL import ImageEnhance
from PIL.Image import Image


class AugBrightness(object):

    def __call__(self, image: Image) -> Image:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.randint(5, 9) / 10.0
        return enhancer.enhance(factor)
