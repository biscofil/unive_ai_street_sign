import csv
import json
import os
import random

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.dataset.ImageAsset import ImageAsset


# https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
class GTSRBDataset(Dataset):
    NO_STREET_SIGN_LABEL_IDX: int = 43

    def __init__(self,
                 transformation_cascade=None,
                 augmentation_operations=None,
                 negative_augmentation_operations=None,
                 test: bool = False):
        self.test: bool = test
        self.transformation_cascade: Compose = transformation_cascade
        self.positive_augmentation: list = [] if augmentation_operations is None else augmentation_operations
        self.negative_augmentation_operations: list = [] if negative_augmentation_operations is None \
            else negative_augmentation_operations

        """self.mult: int = 1
        for op in self.transformation_cascade.transforms:
            if isinstance(op, Augmentation):
                self.mult = self.mult * len(op.augmentation_operations)
        print(self.mult)
        exit(1)"""

        # print("GTSRBDataset will use",
        #       len(self.positive_augmentation),
        #       "augmentation operations on the images")

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
            #
            'no_street_sign'
        ]

    def get_street_sign_images(self) -> list:
        out = []

        base_path: str = os.path.dirname(__file__) + "/GTSRB/Final_{}/Images".format(
            "Test" if self.test else "Training")

        # positive examples: 43 street signs
        for root, dirs, files in os.walk(base_path):
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
        return out

    @staticmethod
    def get_not_street_sign_json_list(test: bool):
        return os.path.dirname(__file__) + "/GTSRB_Negative/images_{}.json".format(
            "test" if test else "training")

    def get_not_street_sign_images(self, random_images: int = 1000) -> list:

        out = []

        with open(self.get_not_street_sign_json_list(self.test), 'r') as fp:
            negative_filenames = json.load(fp)

        base_path: str = os.path.dirname(__file__) + "/GTSRB_Negative/images/"
        n_negative_examples = 0
        for negative_filename in negative_filenames:

            # image without augmentation
            out.append(ImageAsset(base_path + negative_filename,
                                  class_id=GTSRBDataset.NO_STREET_SIGN_LABEL_IDX,
                                  cropping_rect=None))
            n_negative_examples += 1

            # image duplicates with augmentation
            for transformation_list in self.negative_augmentation_operations:
                n_negative_examples += 1
                out.append(ImageAsset(base_path + negative_filename,
                                      augmentation_operations=transformation_list,
                                      class_id=GTSRBDataset.NO_STREET_SIGN_LABEL_IDX,
                                      cropping_rect=None))

        for i in range(0, random_images):  # random noise
            out.append(ImageAsset("",
                                  augmentation_operations=[],
                                  class_id=GTSRBDataset.NO_STREET_SIGN_LABEL_IDX,
                                  cropping_rect=None,
                                  random_noise=True))
        return out

    def get_images(self) -> list:
        out = []

        # positive examples: 43 street signs
        street_sign_images = self.get_street_sign_images()
        out += street_sign_images

        # negative examples: load from folder + random noise + chaotic transformations on real street signs
        neg_e = self.get_not_street_sign_images()
        out += neg_e

        print("\ttest :" if self.test else "\ttrain :",
              len(street_sign_images), "street signs and",
              len(neg_e), "negative examples")

        return out

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):

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
                                         negative_augmentation_operations):

        training_set = GTSRBDataset(
            transformation_cascade=transformation_cascade,
            augmentation_operations=augmentation_operations,
            negative_augmentation_operations=negative_augmentation_operations)

        test_set = GTSRBDataset(
            test=True,
            transformation_cascade=transformation_cascade,
            augmentation_operations=augmentation_operations,
            negative_augmentation_operations=negative_augmentation_operations)

        return training_set, test_set

    @staticmethod
    def add_images_to_negative_examples_json(filenames: list, overwrite: bool, test: bool):

        negative_json = GTSRBDataset.get_not_street_sign_json_list(test)

        if not overwrite:
            if os.path.isfile(negative_json):
                with open(negative_json, 'r') as fp:
                    filenames += json.load(fp)

        filenames = list(dict.fromkeys(filenames))  # remove duplicates
        random.shuffle(filenames)

        if os.path.isfile(negative_json):
            os.remove(negative_json)

        print("Saving {} negative examples to {}".format(len(filenames), negative_json))

        with open(negative_json, 'w') as fp:
            json.dump(filenames, fp)
