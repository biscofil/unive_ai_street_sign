class CustomCrop(object):

    def __call__(self, sample: dict) -> dict:

        image = sample['image'].crop(sample['rect'])

        return {
            "image": image
        }
