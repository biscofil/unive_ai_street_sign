from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import ToPILImage


class DebugImage(object):

    def __init__(self, filename: str, close_program: bool = False):
        self.filename = filename
        self.close_program = close_program

    def __call__(self, pil_image):

        cp = pil_image

        if isinstance(cp, Tensor):
            tr = ToPILImage()
            cp: Image = tr(pil_image)

        if isinstance(cp, Image):
            print("Saving debug image to", self.filename)
            cp.save(self.filename)
        else:
            print("Error in DebugImage: not an image")
            exit(1)

        if self.close_program:
            print("Exiting : DebugImage")
            exit(1)

        return pil_image
