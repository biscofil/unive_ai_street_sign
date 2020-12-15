import numpy
from PIL import Image
from matplotlib.colors import rgb_to_hsv


class Rgb2Hsl:

    def __call__(self, im: Image):
        npi = numpy.array(im)
        npi = rgb_to_hsv(npi)
        npi[:, :, 0] = npi[:, :, 0] * 255
        npi[:, :, 1] = npi[:, :, 1] * 255
        return Image.fromarray(npi.astype(numpy.uint8))
