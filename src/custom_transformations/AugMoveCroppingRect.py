import random

from PIL import Image


class AugMoveCroppingRect:

    def __init__(self, max_shift: int = 2):
        self.max_shift = max_shift

    def __call__(self, sample: dict) -> dict:
        if sample['rect'] is not None:
            image: Image = sample['image']
            left, upper, right, lower = sample['rect']

            left = max(0, left + random.randint(-self.max_shift, self.max_shift))
            upper = max(0, upper + random.randint(-self.max_shift, self.max_shift))

            right = min(image.size[0] - 1, right + random.randint(-self.max_shift, self.max_shift))
            lower = min(image.size[1] - 1, lower + random.randint(-self.max_shift, self.max_shift))

            sample['rect'] = (left, upper, right, lower)

        return sample
