import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import ToTensor

from src.custom_transformations.AugBrightness import AugBrightness
from src.custom_transformations.AugColor import AugColor
from src.custom_transformations.AugMoveCroppingRect import AugMoveCroppingRect
from src.custom_transformations.AugRotate import AugRotate
from src.custom_transformations.CustomCrop import CustomCrop
from src.custom_transformations.CustomRescale import CustomRescale
from src.custom_transformations.NormalizeHist import NormalizeHist
from src.dataset.GTSRBDataset import GTSRBDataset
from src.networks.Trainer import Trainer
from src.networks.utils.NNTrainLoadSave import NNTrainLoadSave


class CnnClassifier(NNTrainLoadSave):
    PATCH_SIZE: int = 30

    def __init__(self, device_name: str):
        super().__init__('classifier.pt')

        self.device_name = device_name
        self.device = torch.device(device_name)

        fe_conv_a_channels = 30  # features
        fe_conv_b_channels = 40  # features
        fe_conv_c_channels = 50  # features

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, fe_conv_a_channels, 3, padding=1),
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

    def forward(self, x: Tensor):
        x = self.feature_extractor(x)
        n, c, h, w = x.shape
        x = x.view(n, -1)  # flat n vectors
        x = self.classifier(x)
        return x

    def train_model(self, learning_rate=0.5, momentum=0.9, batch_size: int = 2000, weight_decay=0.0001,
                    epochs: int = 25):
        print("Training CnnClassifier model...")
        print("\tImg size :", self.PATCH_SIZE)

        transformation_cascade = transforms.Compose([
            #  ---> {"image": PIL, "rect": rect} --->
            AugMoveCroppingRect(),  # ---> {"image": PIL, "rect": rect} --->
            CustomCrop(),  # ---> cropped PIL --->
            CustomRescale(self.PATCH_SIZE),  # ---> square img_size x img_size PIL --->
            NormalizeHist(),  # ---> PIL --->  # with :97%, without: 96-97%
            # TODO Rgb2Hsl
            transforms.ToTensor(),  # ---> Tensor --->
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

    def get_image_label(self, img: Image, resize: bool = True) -> (list, list):
        if resize:
            # scale
            cs = CustomRescale(self.PATCH_SIZE)
            img = cs(img)
        # grayscale
        # window = (CustomToGrayScale())(window)
        # get tensor
        image_tensor = (ToTensor())(img).float()
        image_tensor = image_tensor.unsqueeze_(0)  # .reshape(-1)
        # fix
        # image_tensor = (PreFixBrightnessContrast())(image_tensor)
        return self.get_tensor_label(image_tensor)

    def get_tensor_label(self, tensor: Tensor) -> (list, list):
        # run model
        tensor = tensor.to(self.device)
        tensor.requires_grad = False
        return self.get_variable_label(tensor)

    def get_variable_label(self, nn_input: Tensor) -> (list, list):
        self.eval()
        self.zero_grad()
        output = self(nn_input)
        self.zero_grad()
        dist = (nn.Softmax(dim=1))(output).detach()#dim=1
        del output
        #return dist.data.cpu().numpy().argmax(axis=1), dist.data.cpu().numpy().max(axis=1)
        # return dist.data.cpu().numpy().argmax(axis=1), dist.data.cpu().numpy().max(axis=1)
        return torch.argmax(dist, dim=1).cpu().tolist(), (torch.max(dist, dim=1)[0]).cpu().tolist()