from pathlib import Path

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, model: nn.Module, device, train_loader, test_loader, criterion, optimizer):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.softmax = nn.Softmax(dim=1)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_filename = 'model.zip'

        my_file = Path(self.output_filename)
        if my_file.is_file():
            # file exists
            self.load()

    def train(self, epochs=10):

        print('Initial accuracy: ' + str(self.evaluate()))

        for epoch in range(epochs):
            self.model.train()  # set the model.json to training mode

            for images, labels in self.train_loader:
                self.optimizer.zero_grad()  # don't forget this line!
                images, labels = images.to(self.device), labels.to(self.device)

                output = self.softmax(self.model(images))
                loss = self.criterion(output, labels)
                loss.backward()  # compute the derivatives of the model.json
                self.optimizer.step()  # update weights according to the optimizer

            print('Accuracy at epoch {}: {}'.format(epoch + 1, self.evaluate()))

    def evaluate(self):
        self.model.eval()  # set the model.json to eval mode
        total = 0

        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.softmax(self.model(images))
            predicted = torch.max(output, dim=1)[1]  # argmax the output
            total += (predicted == labels).sum().item()

        return total / len(self.test_loader.dataset)

    def load(self):
        print("Loading model")
        self.model.load_state_dict(torch.load(self.output_filename))
        self.model.eval()

    def save(self):
        print("Storing model")
        torch.save(self.model.state_dict(), self.output_filename)
