from PIL import Image


class Augmentation(object):

    def __init__(self, augmentation_operations : list):
        self.augmentation_operations = augmentation_operations

    def __call__(self, image: Image) -> Image:
        # transform
        augmentation_callbacks = self.augmentation_operations
        if not isinstance(augmentation_callbacks, list):
            augmentation_callbacks = [augmentation_callbacks]
        for callback in augmentation_callbacks:
            if callback is not None:
                # call augmentation callback
                Image = callback(Image)
        return image
