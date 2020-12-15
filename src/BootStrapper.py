import json
import os
import random

from PIL import Image

from src.dataset.GTSRBDatasetWithNegatives import GTSRBDatasetWithNegatives
from src.networks.CnnDetector import CnnDetector


class BootStrapper:

    @staticmethod
    def populate_json_few(limit: int = 1000):
        print("Bootstrapping")
        random_images = []
        folder = os.path.dirname(__file__) + '/dataset/GTSRB/Negative/images/'
        print("Searching for a few pictures in", folder)
        for root, dirs, files in os.walk(folder):
            for name in files:
                if name.endswith(".jpg"):
                    random_images.append(name)
        print(len(random_images), "will be used as negative examples")
        random.shuffle(random_images)
        random_images = random_images[:limit]
        GTSRBDatasetWithNegatives.add_images_to_negative_examples_json(random_images, overwrite=True)

    @staticmethod
    def get_new_jpgs() -> list:
        folder = os.path.dirname(__file__) + '/dataset/GTSRB/Negative/images/'
        print("Searching for new pictures in", folder, "that are classified as street signs")

        # load from json file in order to ignore already labeled as negative
        with open(GTSRBDatasetWithNegatives.NEGATIVE_JSON, 'r') as fp:
            negative_filenames = json.load(fp)

        jpg_files = []
        for root, dirs, files in os.walk(folder):
            for name in files:
                if name.endswith(".jpg") and name not in negative_filenames:
                    jpg_files.append((root, name))

        return jpg_files

    @staticmethod
    def add_wrongly_classified_to_negative_json(model: CnnDetector, limit: int = 51000):
        print("Bootstrapping")
        wrongly_classified = []

        jpg_files = BootStrapper.get_new_jpgs()
        print(len(jpg_files), "new files")

        random.shuffle(jpg_files)
        for (root, name) in jpg_files:
            if len(wrongly_classified) > limit:
                break
            full_filepath = root + name
            try:
                im = Image.open(full_filepath)
            except:
                print("Error loading ", full_filepath)
                continue

            label_idx,_ = model.get_image_label(im)
            if not label_idx == GTSRBDatasetWithNegatives.STREET_SIGN_LABEL_IDX:
                wrongly_classified.append(name)

        print(len(wrongly_classified), "were wrongly classified. They will be used as negative examples")

        random.shuffle(wrongly_classified)
        wrongly_classified = wrongly_classified[:limit]
        GTSRBDatasetWithNegatives.add_images_to_negative_examples_json(wrongly_classified, overwrite=False)
