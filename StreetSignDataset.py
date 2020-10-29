import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class StreetSignDataset(Dataset):

    def __init__(self, base_path="./GTSRB-Training_fixed/GTSRB/Training", transform=None):
        self.transform = transform
        self.base_path = base_path
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
        print(len(labels), "labels")
        print(labels)
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
