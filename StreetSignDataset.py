import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


# http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset

class StreetSignDataset(Dataset):

    def __init__(self, base_path="./GTSRB-Training_fixed/GTSRB/Training", transform=None):
        self.transform = transform
        self.base_path = base_path
        self.label_names = [
            '20kmh',  # 0000
            '30kmh',
            '50kmh',
            '60kmh',
            '70kmh',
            '80kmh',
            'end_80kmh',
            '100kmh',
            '120kmh',
            'no_passing',
            'no_passing_for_vehicles_3mt+',  # 0010
            'intersection',
            'priority_road',
            'yield',
            'stop',
            'no_vehicles',
            'no_vehicle_3mt+',
            'no_entry',
            'general_caution',
            'dangerous_left_curve',
            'dangerous_right_curve',  # 0020
            'double_curve',
            'bumpy_road',
            'slippery_road',
            'left_narrowing',
            'man_at_work',
            'traffic_light',
            'zebra_crossing',
            'children_crossing',
            'bicycle_crossing',
            'ice_snow',  # 0030
            'wild_animals',
            'end_speed_limit',
            'turn_right_ahead',
            'turn_left_ahead',
            'ahead_only',
            'straight_or_right',
            'straight_or_left',
            'keep_right',
            'keep_left',
            'roundabout',  # 0040
            'end_no_passing',
            'end_no_passing_for_vehicles_3mt+'
        ]
        self.images = self.get_images()

    def get_images(self):
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
        assert (len(self.label_names) == len(labels))
        return out

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        image_filename = image["Filename"]
        image_data = Image.open(image_filename)

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
