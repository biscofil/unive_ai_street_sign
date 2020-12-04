import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.custom_transformations.CustomCrop import CustomCrop
from src.custom_transformations.CustomRescale import CustomRescale
from src.dataset.GTSRBStreetSignDataset import GTSRBStreetSignDataset
from src.networks.Evaluator import Evaluator


class Trainer(Evaluator):

    def __init__(self, model: nn.Module, device: str, img_size: int,
                 learning_rate, momentum, batch_size: int, weight_decay: float):
        super().__init__(model, device)

        self.criterion = nn.CrossEntropyLoss()

        # stochastic gradient descent
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay)

        transform = transforms.Compose([
            # {"image": PIL, "rect": rect}
            CustomCrop(),
            # cropped PIL
            CustomRescale(img_size),
            # square img_size x img_size PIL
            # CustomToGrayScale(),
            # grayscale square img_size x img_size PIL
            transforms.ToTensor()
            # Tensor
        ])

        ssd = GTSRBStreetSignDataset(transform=transform, use_augmentation=False)
        train_dataset, test_dataset = ssd.get_training_testing_datasets()

        """for i in range(100, 130):
            im, label = ssd.__getitem__(i)
            im.save("img_" + str(i) + ".png", "JPEG")
        exit(1)"""

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs: int) -> None:

        print('Initial accuracy: {:06.4f}'.format(self.evaluate()))

        for epoch in range(epochs):
            self.model.train()  # enter training mode

            start = time.time()
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()  # don't forget this line!
                labels = labels.to(self.device)
                output = super().evaluate_images(images)  # get softmax output
                loss = self.criterion(output, labels)
                loss.backward()  # compute the derivatives of the model
                self.optimizer.step()  # update weights according to the optimizer
            training_duration_sec = int(time.time() - start)

            start = time.time()
            e = self.evaluate()
            test_duration_sec = int(time.time() - start)

            print('Accuracy at epoch {}: {:06.4f} trained in {:d}s, evaluated in {:d}s'.format(epoch + 1, e,
                                                                                               training_duration_sec,
                                                                                               test_duration_sec))

    def evaluate(self):
        self.model.eval()  # enter evaluation mode
        total = 0

        for images, labels in self.test_loader:
            labels = labels.to(self.device)
            output = super().evaluate_images(images)  # get softmax output
            predicted = torch.max(output, dim=1)[1]
            total += (predicted == labels).sum().item()

        return total / len(self.test_loader.dataset)
