import csv
import os

from torch.utils.data import Dataset, random_split
# https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
from torchvision.transforms import Compose

from src.custom_transformations.Augmentation import Augmentation
from src.dataset.ImageAsset import ImageAsset


class GTSRBDataset(Dataset):
    CLASS_ID_DICT_KEY: str = "ClassId"

    def __init__(self,
                 transformation_cascade=None,
                 augmentation_operations=None,
                 evaluation: bool = False):
        self.evaluation: bool = evaluation

        self.base_path: str = os.path.dirname(__file__) + "/GTSRB/Final_{}/Images".format(
            "Test" if evaluation else "Training")  # TODO check!!!!!

        self.transformation_cascade: Compose = transformation_cascade
        self.positive_augmentation: list = [] if augmentation_operations is None else augmentation_operations

        """self.mult: int = 1
        for op in self.transformation_cascade.transforms:
            if isinstance(op, Augmentation):
                self.mult = self.mult * len(op.augmentation_operations)
        print(self.mult)
        exit(1)"""

        print("GTSRBDataset will use",
              len(self.positive_augmentation),
              "augmentation operations on the images")

        self.images = self.get_images()

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
            'end_no_passing', 'end_no_passing_for_vehicles_3mt+',
        ]

    def get_images(self) -> list:
        out = []

        # positive examples: 43 street signs
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                if name.endswith(".csv"):
                    full_path = root + "/" + name
                    with open(full_path) as csv_file:
                        csv_reader = csv.DictReader(csv_file, delimiter=';', quoting=csv.QUOTE_NONE)
                        csv_image_rows = list(csv_reader)
                    for csv_image_row in csv_image_rows:
                        # image without augmentation
                        out.append(ImageAsset.from_gtsrb_csv_image_record(csv_image_row, root))
                        # image duplicates with augmentation
                        for transformation_list in self.positive_augmentation:
                            out.append(ImageAsset.from_gtsrb_csv_image_record(csv_image_row, root, transformation_list))

        print("\ttest :" if self.evaluation else "\ttrain :", len(out), "positive examples")

        return out

    def __len__(self) -> int:
        return len(self.images)  # TODO * self.mult

    def __getitem__(self, idx: int):
        # TODO idx = idx %  len(self.images)

        # called A LOT
        image: ImageAsset = self.images[idx]

        out = {
            "image": image.load_and_transform_image(),
            "rect": image.cropping_rect
        }

        if self.transformation_cascade:
            out = self.transformation_cascade(out)

        return out, image.class_id

    @staticmethod
    def get_training_evaluation_datasets(transformation_cascade, augmentation_operations,
                                         training_fraction: float = 0.7):  # TODO check

        if False:
            full_dataset = GTSRBDataset(
                transformation_cascade=transformation_cascade,
                augmentation_operations=augmentation_operations)

            n: int = len(full_dataset)
            n_train_img: int = int(n * training_fraction)
            n_val_img: int = n - n_train_img

            return random_split(full_dataset, (n_train_img, n_val_img))

        else:

            training_set = GTSRBDataset(
                transformation_cascade=transformation_cascade,
                augmentation_operations=augmentation_operations)

            test_set = GTSRBDataset(
                evaluation=True,
                transformation_cascade=transformation_cascade,
                augmentation_operations=augmentation_operations)

            return training_set, test_set
