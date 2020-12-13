import torch
import torch.nn as nn
from torchvision import transforms

from src.custom_transformations.AugBrightness import AugBrightness
from src.custom_transformations.AugColor import AugColor
from src.custom_transformations.AugMoveCroppingRect import AugMoveCroppingRect
from src.custom_transformations.AugRotate import AugRotate
from src.custom_transformations.Bypass import Bypass
from src.custom_transformations.CustomCrop import CustomCrop
from src.custom_transformations.CustomRescale import CustomRescale
from src.custom_transformations.CustomToGrayScale import CustomToGrayScale
from src.dataset.GTSRBDataset import GTSRBDataset
from src.networks.Trainer import Trainer
from src.networks.utils.NNTrainLoadSave import NNTrainLoadSave


class CnnClassifier(NNTrainLoadSave):
    IMG_SIZE: int = 30

    def __init__(self, device_name: str, use_rgb: bool = True):
        super().__init__('classifier.pt')

        self.device_name = device_name
        self.device = torch.device(device_name)
        self.use_rgb = use_rgb

        fe_conv_a_channels = 30  # features
        fe_conv_b_channels = 40  # features
        fe_conv_c_channels = 50  # features

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3 if use_rgb else 1, fe_conv_a_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(fe_conv_a_channels),
            nn.MaxPool2d(2),

            ########################################

            nn.Conv2d(fe_conv_a_channels, fe_conv_b_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(fe_conv_b_channels),
            nn.MaxPool2d(2),

            ########################################

            nn.Conv2d(fe_conv_b_channels, fe_conv_c_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(fe_conv_c_channels),

        )

        n2: int = 200
        n3: int = 200
        output_size: int = len(GTSRBDataset.get_label_names())

        self.classifier = nn.Sequential(
            nn.Linear(800, n2),
            nn.ReLU(),
            nn.BatchNorm1d(n2),

            ########################################

            nn.Linear(n2, n3),
            nn.ReLU(),
            nn.BatchNorm1d(n3),

            ########################################

            nn.Linear(n3, output_size),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        n, c, h, w = x.shape
        x = x.view(n, -1)

        if False:  # TODO remove
            print(x.shape)
            exit(1)

        try:
            x = self.classifier(x)
        except:
            print("feature extractor returns unexpected structure with shape", x.shape)
            exit(1)
        return x

    def train_model(self, learning_rate=0.5, momentum=0.9, batch_size: int = 2000, weight_decay=0.0001,
                    epochs: int = 25):
        print("Training CnnClassifier model...")
        print("\tImg size :", self.IMG_SIZE)

        transformation_cascade = transforms.Compose([
            #  ---> {"image": PIL, "rect": rect} --->
            AugMoveCroppingRect(),  # ---> {"image": PIL, "rect": rect} --->
            CustomCrop(),  # ---> cropped PIL --->
            CustomRescale(self.IMG_SIZE),  # ---> square img_size x img_size PIL --->
            # NormalizeHist(),  # ---> PIL --->  # with :97%, without: 96-97%
            Bypass() if self.use_rgb else CustomToGrayScale(),  # ---> grayscale PIL --->
            transforms.ToTensor(),  # ---> Tensor --->
            # FixTensorBrightnessContrast(),  # ---> Tensor --->
        ])

        augmentation_operations = [
            [
                AugRotate(random_factor=True, r_min=-1, r_max=1),
                AugColor(random_factor=True, r_min=3, r_max=7),
                AugBrightness(random_factor=True, r_min=3, r_max=7)
            ],
            [
                AugRotate(random_factor=True, r_min=-1, r_max=1),
                AugColor(random_factor=True, r_min=3, r_max=7),
                AugBrightness(random_factor=True, r_min=3, r_max=7)
            ]
        ]

        train_dataset, validation_dataset = GTSRBDataset.get_training_evaluation_datasets(
            transformation_cascade=transformation_cascade,
            augmentation_operations=augmentation_operations)  # TODO check

        trainer = Trainer(self, self.device_name,
                          learning_rate, momentum, batch_size, weight_decay,
                          train_dataset, validation_dataset,
                          nn.CrossEntropyLoss())

        trainer.train(epochs)
