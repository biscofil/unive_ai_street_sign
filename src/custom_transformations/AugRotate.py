import random

from PIL.Image import Image


class AugRotate(object):

    def __init__(self, angle: int = 90, random_factor: bool = False, r_min: int = 0, r_max: int = 10):
        self.angle: int = angle
        self.random_factor: bool = random_factor
        self.r_min: int = r_min
        self.r_max: int = r_max

    def __call__(self, image: Image) -> Image:
        if self.random_factor:
            self.angle = random.randint(self.r_min, self.r_max)
        return image.rotate(self.angle)
