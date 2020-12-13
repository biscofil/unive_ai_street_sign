from PIL import Image


class ImageAsset:

    @staticmethod
    def from_gtsrb_csv_image_record(csv_img_row: dict, folder_root: str,
                                    augmentation_operations=None, use_crop_rect: bool = True):

        return ImageAsset(

            folder_root + "/" + csv_img_row["Filename"],

            augmentation_operations=augmentation_operations,

            # cropping rectangle
            cropping_rect=(
                int(csv_img_row["Roi.X1"]), int(csv_img_row["Roi.Y1"]),
                int(csv_img_row["Roi.X2"]), int(csv_img_row["Roi.Y2"]))
            if use_crop_rect else None,

            # class
            class_id=int(csv_img_row["ClassId"])
        )

    def __init__(self, filename: str, cropping_rect, class_id: int, augmentation_operations: list = None):
        self.filename: str = filename
        self.augmentation_operations: list = [] if augmentation_operations is None else augmentation_operations
        # must have format X1,Y1,X2,Y2
        self.cropping_rect = cropping_rect
        self.class_id: int = class_id

    def print(self):
        print("Image:")
        print("\tFilename:", self.filename)
        print("\tCropping rect:", self.cropping_rect)
        print("\tClass:", self.class_id)

    def load_and_transform_image(self) -> Image:
        # load image
        image_filename = self.filename
        image_data = Image.open(image_filename).convert('RGB')
        # transform
        augmentation_callbacks = self.augmentation_operations
        if not isinstance(augmentation_callbacks, list):
            augmentation_callbacks = [augmentation_callbacks]
        for callback in augmentation_callbacks:
            # call augmentation callback
            image_data = callback(image_data)
        return image_data
