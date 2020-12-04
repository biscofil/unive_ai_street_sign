import random

from PIL.Image import Image


class AugRotate(object):

    def __call__(self, image: Image) -> Image:
        return image.rotate(random.randint(-3, 3))
