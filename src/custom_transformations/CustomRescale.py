from PIL import Image


class CustomRescale(object):

    def __init__(self, output_size: int):
        self.output_size: int = output_size

    def __call__(self, img: Image) -> Image:
        return img.resize((self.output_size, self.output_size))
