from pathlib import Path

import torch
import torch.nn as nn

from src.dataset.GTSRBStreetSignDataset import GTSRBStreetSignDataset
from src.networks.Trainer import Trainer


class FullNetwork(nn.Module):

    def __init__(self, device_name: str):
        super().__init__()
        self.img_size: int = 30

        self.device_name = device_name
        self.device = torch.device(device_name)

        fe_conv_a_channels = 30  # features
        fe_conv_b_channels = 40  # features
        fe_conv_c_channels = 50  # features
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, fe_conv_a_channels, 3),
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
            nn.MaxPool2d(2)
        )

        n2: int = 200
        n3: int = 200
        output_size: int = len(GTSRBStreetSignDataset.get_label_names())  # +1

        self.classifier = nn.Sequential(
            nn.Linear(200, n2),
            nn.ReLU(),
            nn.BatchNorm1d(n2),

            ########################################

            nn.Linear(n2, n3),
            nn.ReLU(),
            nn.BatchNorm1d(n3),

            ########################################

            nn.Linear(n3, output_size),
            # nn.ReLU()
        )

        self.output_filename = 'network.zip'
        self.load()

    def forward(self, x):
        x = self.feature_extractor(x)
        n, c, h, w = x.shape
        x = x.view(n, -1)

        if False:
            print(x.shape)
            exit(1)

        x = self.classifier(x)
        return x

    def train_model(self, learning_rate=0.1, momentum=0.9, batch_size: int = 80, weight_decay=0.0001, epochs: int = 30):
        print("Training model...")
        print("\tImg size :", self.img_size)
        print("\tLearning Rate :", learning_rate)
        print("\tMomentum :", momentum)
        print("\tBatch Size :", batch_size)
        print("\tWeight Decay :", weight_decay)
        trainer = Trainer(self, self.device_name, self.img_size, learning_rate, momentum, batch_size, weight_decay)
        trainer.train(epochs)
        #self.save()

    def load(self):
        my_file = Path(self.output_filename)
        if my_file.is_file():
            # file exists
            try:
                print("Loading model...")
                self.load_state_dict(torch.load(self.output_filename))
                self.eval()
                self.to(self.device_name)
            except:
                print("Loading failed! Did the structure change?")

    def save(self):
        print("Storing model...")
        torch.save(self.state_dict(), self.output_filename)
