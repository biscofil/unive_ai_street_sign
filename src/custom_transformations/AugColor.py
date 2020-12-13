import random

from PIL import ImageEnhance
from PIL.Image import Image


class AugColor(object):

    def __init__(self, factor: float = 0.5, random_factor: bool = False, r_min: int = 0, r_max: int = 10):
        self.factor: float = factor
        self.random_factor: bool = random_factor
        self.r_min: int = r_min
        self.r_max: int = r_max

    def __call__(self, image: Image) -> Image:
        if self.random_factor:
            self.factor = random.randint(self.r_min, self.r_max) / 10.0
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(self.factor)
