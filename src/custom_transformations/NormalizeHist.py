from PIL import ImageOps
from PIL.Image import Image


class NormalizeHist(object):

    def __call__(self, pil_image: Image) -> Image:
        pil_image = ImageOps.autocontrast(pil_image)
        pil_image = ImageOps.equalize(pil_image)
        return pil_image
