from PIL import ImageOps
from PIL.Image import Image


class AugMirror(object):

    def __call__(self, image: Image) -> Image:
        return ImageOps.mirror(image)
