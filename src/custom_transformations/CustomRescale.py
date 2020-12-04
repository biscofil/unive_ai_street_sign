from PIL import Image


class CustomRescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample: dict) -> Image:
        new_h, new_w = self.output_size, self.output_size

        img = sample["image"]

        img = img.resize((new_h, new_w))

        return img
