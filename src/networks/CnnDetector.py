import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor

from src.custom_transformations.AugBrightness import AugBrightness
from src.custom_transformations.AugColor import AugColor
from src.custom_transformations.AugFlip import AugFlip
from src.custom_transformations.AugMirror import AugMirror
from src.custom_transformations.AugMoveCroppingRect import AugMoveCroppingRect
from src.custom_transformations.AugRotate import AugRotate
from src.custom_transformations.Bypass import Bypass
from src.custom_transformations.CustomCrop import CustomCrop
from src.custom_transformations.CustomRescale import CustomRescale
from src.custom_transformations.CustomToGrayScale import CustomToGrayScale
from src.dataset.GTSRBDatasetWithNegatives import GTSRBDatasetWithNegatives
from src.networks.Trainer import Trainer
from src.networks.utils.NNTrainLoadSave import NNTrainLoadSave


class CnnDetector(NNTrainLoadSave):
    IMG_SIZE: int = 28

    def __init__(self, device_name: str, use_rgb: bool = True):
        super().__init__('detector.pt')

        self.device_name = device_name
        self.device = torch.device(device_name)
        self.use_rgb = use_rgb

        fe_conv_a_channels = 15  # features
        fe_conv_b_channels = 30  # features
        fe_conv_c_channels = 45  # features

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(3 if self.use_rgb else 1, fe_conv_a_channels, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(fe_conv_a_channels),
            nn.MaxPool2d(2),

            ########################################

            nn.Conv2d(fe_conv_a_channels, fe_conv_b_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(fe_conv_b_channels),
            nn.MaxPool2d(2),

            #########################################

            nn.Conv2d(fe_conv_b_channels, fe_conv_c_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(fe_conv_c_channels),

        )

        n2: int = 200
        n3: int = 200
        output_size: int = len(GTSRBDatasetWithNegatives.get_label_names())

        self.classifier = nn.Sequential(
            nn.Linear(720, n2),
            nn.ReLU(),
            nn.BatchNorm1d(n2),

            ########################################

            nn.Linear(n2, n3),
            nn.ReLU(),
            nn.BatchNorm1d(n3),

            ########################################

            nn.Linear(n3, output_size),  # boolean
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
                    epochs: int = 10) -> None:

        print("Training CnnDetector model...")
        print("\tImg size :", self.IMG_SIZE)

        pos_transformation_cascade = transforms.Compose([
            # --> {"image": PIL, "rect": rect} -->
            AugMoveCroppingRect(),  # --> {"image": PIL, "rect": rect} -->
            CustomCrop(),  # --> cropped PIL -->
            CustomRescale(self.IMG_SIZE),  # --> square img_size x img_size PIL -->
            # DebugImage('before_norm.png'),
            # NormalizeHist(),  # TODO BAD
            Bypass() if self.use_rgb else CustomToGrayScale(),  # --> square grayscale img_size x img_size PIL -->
            transforms.ToTensor(),  # --> Tensor -->
            # FixTensorBrightnessContrast(),  # TODO BAD
            # torchvision.transforms.Normalize(mean=0.5, std=0.2),
            # DebugImage('after_norm.png', exit=True),
        ])

        neg_transformation_cascade = transforms.Compose([
            # {"image": PIL, "rect": rect}
            CustomCrop(),  # --> cropped PIL -->
            CustomRescale(self.IMG_SIZE),  # --> square img_size x img_size PIL -->
            # Augmentation([
            #     None,
            #     [AugRotate(90), AugColor(random.randint(3, 7) / 10.0)],
            # ]),
            # DebugImage('before_norm.png'),
            # NormalizeHist(),
            # CustomToGrayScale(),  # --> square grayscale img_size x img_size PIL -->
            transforms.ToTensor(),  # --> Tensor -->
            # FixTensorBrightnessContrast(),
            # torchvision.transforms.Normalize(mean=0.5, std=0.2),
            # DebugImage('after_norm.png', exit=True),
            # TODO for negative ShuffleTensor(),
        ])

        # negative_augmentation_operations = [
        # [AugRotate(random.randint(10, 45)), AugColor(random.randint(3, 7) / 10.0)],
        # [AugMirror(), AugFlip()],
        # AugMirror(),
        # AugFlip(),
        # AugRotate(90),  # rotate
        # AugRotate(-90),  # rotate
        # AugBrightness(random.randint(3, 7) / 10.0),  # change brightness
        # AugBrightness(0.5),  # change brightness
        # AugColor(random.randint(3, 7) / 10.0),  # change color
        # AugColor(0.5),  # change color
        # [AugRotate(-20), AugColor(0.7)],  # rotate then change brightness
        # [AugRotate(20), AugBrightness(0.7)],  # rotate then change brightness
        # ]

        train_dataset, evaluation_dataset = GTSRBDatasetWithNegatives.get_neg_training_evaluation_datasets(
            transformation_cascade=pos_transformation_cascade,
            positive_augmentation_operations=[
                # NO ROTATION BEFORE CROPPING
                AugRotate(random_factor=True, r_min=-2, r_max=2),
                AugColor(random_factor=True, r_min=3, r_max=7),
                AugBrightness(random_factor=True, r_min=3, r_max=7)
            ],
            negative_augmentation_operations=[
                # AugRotate(random_factor=True, r_min=1, r_max=4),
                # AugRotate(random_factor=True, r_min=-4, r_max=-1),
                AugMirror(),
                AugFlip(),
                AugRotate(90),
                AugRotate(180),
                AugRotate(270),
                AugColor(random_factor=True, r_min=1, r_max=9),
                AugBrightness(random_factor=True, r_min=1, r_max=9)
            ]
        )

        trainer = Trainer(self, self.device_name, learning_rate, momentum, batch_size, weight_decay,
                          train_dataset, evaluation_dataset,
                          nn.CrossEntropyLoss())
        trainer.train(epochs)

    def get_image_label(self, img: Image, resize: bool = True) -> int:
        if resize:
            # scale
            cs = CustomRescale(self.IMG_SIZE)
            img = cs(img)
        # grayscale
        # window = (CustomToGrayScale())(window)
        # get tensor
        image_tensor = (ToTensor())(img).float()
        image_tensor = image_tensor.unsqueeze_(0)  # .reshape(-1)
        # fix
        # image_tensor = (PreFixBrightnessContrast())(image_tensor)
        return self.get_tensor_label(image_tensor)

    def get_tensor_label(self, tensor: Tensor) -> int:
        # run model
        nn_input = Variable(tensor)
        nn_input = nn_input.to(self.device)
        return self.get_variable_label(nn_input)

    def get_variable_label(self, nn_input) -> int:
        output = self(nn_input)
        dist = (nn.Softmax(dim=1))(output)
        return dist.data.cpu().numpy().argmax()
