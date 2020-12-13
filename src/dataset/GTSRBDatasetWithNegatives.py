import json
import os
import random

import torch
from torch.utils.data import random_split

from src.dataset.GTSRBDataset import GTSRBDataset
from src.dataset.ImageAsset import ImageAsset


# https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html


class GTSRBDatasetWithNegatives(GTSRBDataset):
    NO_STREET_SIGN_LABEL: str = 'no_street_sign'
    NO_STREET_SIGN_LABEL_IDX: int = 0
    STREET_SIGN_LABEL: str = 'street_sign'
    STREET_SIGN_LABEL_IDX: int = 1

    NEGATIVE_JSON: str = os.path.dirname(__file__) + '/GTSRB/Negative/images.json'

    def __init__(self,
                 evaluation: bool = False,
                 transformation_cascade=None,
                 positive_augmentation_operations=None,
                 negative_augmentation_operations=None,
                 train_fraction: float = 0.68):

        self.negative_augmentation_operations: list = [] if negative_augmentation_operations is None \
            else negative_augmentation_operations
        self.train_fraction: float = train_fraction

        super().__init__(evaluation=evaluation,
                         transformation_cascade=transformation_cascade,
                         augmentation_operations=positive_augmentation_operations)

        print("GTSRBDatasetWithNegatives will use",
              len(self.negative_augmentation_operations),
              "augmentation operations on the negative examples")

    @staticmethod
    def get_label_names() -> list:
        labels = [
            GTSRBDatasetWithNegatives.NO_STREET_SIGN_LABEL,  # NO_STREET_SIGN_LABEL_IDX
            GTSRBDatasetWithNegatives.STREET_SIGN_LABEL  # STREET_SIGN_LABEL_IDX
        ]
        return labels

    def __get_images(self) -> list:

        # ############# positive_images
        out = super().get_images()
        n_positive_examples = len(out)
        positive_example: ImageAsset
        for positive_example in out:
            positive_example.class_id = GTSRBDatasetWithNegatives.STREET_SIGN_LABEL_IDX  # street sign

        # ############# negative examples: no_street_signs

        # load from json file
        with open(self.NEGATIVE_JSON, 'r') as fp:
            negative_filenames = json.load(fp)

        # load for training  / test (same amount as positive examples)
        n_train_img = int(len(negative_filenames) * self.train_fraction)
        negative_filenames = negative_filenames[n_train_img:] if self.evaluation else negative_filenames[:n_train_img]
        folder_root = os.path.dirname(__file__) + '/GTSRB/Negative/images/'

        n_negative_examples = 0
        for negative_filename in negative_filenames:

            # image without augmentation
            out.append(ImageAsset(folder_root + negative_filename,
                                  class_id=GTSRBDatasetWithNegatives.NO_STREET_SIGN_LABEL_IDX,
                                  cropping_rect=None))
            n_negative_examples += 1

            # image duplicates with augmentation
            for transformation_list in self.negative_augmentation_operations:
                n_negative_examples += 1
                out.append(ImageAsset(folder_root + negative_filename,
                                      augmentation_operations=transformation_list,
                                      class_id=GTSRBDatasetWithNegatives.NO_STREET_SIGN_LABEL_IDX,
                                      cropping_rect=None))

        print("\ttest :" if self.evaluation else "\ttrain :",
              n_positive_examples, "positive examples and",
              n_negative_examples, "negative examples")

        return out

    def __len__(self):
        # TODO check
        return super().__len__() * 2

    def __getitem__(self, index):
        if index < super().__len__():
            tensor, label = super().__getitem__(index)
            return tensor, self.STREET_SIGN_LABEL_IDX
        else:
            n = 28
            return torch.rand(3, n, n), self.NO_STREET_SIGN_LABEL_IDX

    @staticmethod
    def add_images_to_negative_examples_json(filenames: list, overwrite: bool):

        if not overwrite:
            if os.path.isfile(GTSRBDatasetWithNegatives.NEGATIVE_JSON):
                with open(GTSRBDatasetWithNegatives.NEGATIVE_JSON, 'r') as fp:
                    filenames = filenames + json.load(fp)

        filenames = list(dict.fromkeys(filenames))  # remove duplicates

        random.shuffle(filenames)

        os.remove(GTSRBDatasetWithNegatives.NEGATIVE_JSON)

        print("Saving {} negative examples to {}".format(len(filenames),
                                                         GTSRBDatasetWithNegatives.NEGATIVE_JSON))

        with open(GTSRBDatasetWithNegatives.NEGATIVE_JSON, 'w') as fp:
            json.dump(filenames, fp)

    @staticmethod
    def get_neg_training_evaluation_datasets(transformation_cascade,
                                             positive_augmentation_operations,
                                             negative_augmentation_operations,
                                             training_fraction: float = 0.7):  # TODO check

        # TODO use random images as test set

        full_dataset = GTSRBDatasetWithNegatives(
            transformation_cascade=transformation_cascade,
            positive_augmentation_operations=positive_augmentation_operations,
            negative_augmentation_operations=negative_augmentation_operations)

        n: int = len(full_dataset)
        n_train_img: int = int(n * training_fraction)
        n_val_img: int = n - n_train_img

        training_dataset, evaluation_dataset = random_split(full_dataset, (n_train_img, n_val_img))

        # TODO training_dataset.setAsTraining()
        # TODO evaluation_dataset.setAsEvaluation()

        return training_dataset, evaluation_dataset
