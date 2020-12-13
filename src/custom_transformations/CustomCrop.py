from PIL import Image


class CustomCrop(object):

    def __call__(self, sample: dict) -> Image:
        image: Image = sample['image']

        if sample['rect'] is not None:
            left, upper, right, lower = sample['rect']
            image = image.crop((left, upper, right, lower))

        return image
