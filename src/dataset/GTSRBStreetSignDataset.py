import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

# http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset
from src.custom_transformations.AugBrightness import AugBrightness
from src.custom_transformations.AugColor import AugColor
from src.custom_transformations.AugRotate import AugRotate


class GTSRBStreetSignDataset(Dataset):

    def __init__(self, transform=None, use_augmentation: bool = False):
        self.transform = transform
        self.base_path = os.path.dirname(__file__) + "/GTSRB-Training_fixed/GTSRB/Training"
        self.images = self.get_images()

        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augmentation_transformations = [
                None,  # do not remove!  use image without transformations
                AugRotate(),  # rotate
                AugBrightness(),  # change brightness
                AugColor(),  # change color
                [AugRotate(), AugColor()],  # rotate then change brightness
                [AugRotate(), AugBrightness()],  # rotate then change brightness
            ]
        else:
            self.augmentation_transformations = [
                None  # do not remove! use image without transformations
            ]

    @staticmethod
    def get_label_names() -> list:
        return [
            '20kmh',  # 0000
            '30kmh', '50kmh', '60kmh', '70kmh', '80kmh', 'end_80kmh', '100kmh', '120kmh',
            'no_passing', 'no_passing_for_vehicles_3mt+',  # 0010
            'intersection',
            'priority_road',
            'yield',
            'stop',
            'no_vehicles', 'no_vehicle_3mt+',
            'no_entry',
            'general_caution',
            'dangerous_left_curve', 'dangerous_right_curve',  # 0020
            'double_curve',
            'bumpy_road',
            'slippery_road',
            'left_narrowing',
            'man_at_work',
            'traffic_light',
            'zebra_crossing', 'children_crossing', 'bicycle_crossing',
            'ice_snow',  # 0030
            'wild_animals',
            'end_speed_limit',
            'turn_right_ahead', 'turn_left_ahead',
            'ahead_only',
            'straight_or_right', 'straight_or_left',
            'keep_right', 'keep_left',
            'roundabout',  # 0040
            'end_no_passing', 'end_no_passing_for_vehicles_3mt+'
        ]

    def get_images(self) -> list:
        out = []
        labels = []
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                if name.endswith(".csv"):
                    full_path = root + "/" + name
                    with open(full_path) as csv_file:
                        csv_reader = csv.DictReader(csv_file, delimiter=';', quoting=csv.QUOTE_NONE)
                        images = list(csv_reader)
                        labels.append(int(images[0]["ClassId"]))
                        for image in images:
                            image["Filename"] = root + "/" + image["Filename"]
                            out.append(image)
        assert (len(self.get_label_names()) == len(labels))
        return out

    def get_image(self, index: int) -> dict:
        img_index = int(index / len(self.augmentation_transformations))
        return self.images[img_index]

    def transform_image(self, image: Image, index: int) -> Image:
        t_index = index % len(self.augmentation_transformations)
        callbacks = self.augmentation_transformations[t_index]
        if callbacks is not None:
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            for callback in callbacks:
                # call transformation
                image = callback(image)
        return image

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset
        """
        return len(self.images) * len(self.augmentation_transformations)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):  # TODO remove
            idx = idx.tolist()
            print("idx is tensor")
            print(idx)

        image = self.get_image(idx)
        image_filename = image["Filename"]
        image_data = Image.open(image_filename)
        # image_data = image_data.convert('1')  #  grayscale

        # augment data by rotating and blurring a little bit
        image_data = self.transform_image(image_data, idx)

        upper = int(image["Roi.Y1"])
        lower = int(image["Roi.Y2"])
        left = int(image["Roi.X1"])
        right = int(image["Roi.X2"])

        out = {
            "image": image_data,
            "rect": (left, upper, right, lower)
        }

        if self.transform:
            out = self.transform(out)

        return out, int(image["ClassId"])

    def get_training_testing_datasets(self, train_fraction=0.8):
        image_count = len(self)
        n_train_img = int(image_count * train_fraction)
        n_test_img = image_count - n_train_img
        return torch.utils.data.random_split(self, [n_train_img, n_test_img])
