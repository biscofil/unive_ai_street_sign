import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Trainer(object):

    def __init__(self, model: nn.Module, device: str,
                 learning_rate: float, momentum: float, batch_size: int, weight_decay: float,
                 train_dataset: Dataset, validation_dataset: Dataset,
                 loss_function):

        self.model: nn.Module = model.to(device)
        self.device: str = device
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.batch_size: int = batch_size

        self.loss_function = loss_function
        self.softmax = nn.Softmax(dim=1)

        # stochastic gradient descent
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                   weight_decay=weight_decay)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs: int) -> None:

        print("\tEpochs :", epochs)
        print("\tLearning Rate :", self.learning_rate)
        print("\tMomentum :", self.momentum)
        print("\tBatch Size :", self.batch_size)
        print("\tWeight Decay :", self.weight_decay)

        accuracy, eval_duration = self.evaluate()
        print('Initial accuracy: {:07.5f}'.format(accuracy))
        summary = {
            "structure": self.model.__str__(),
            "epochs": epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "training": [{
                "accuracy": accuracy,
                "test_duration": eval_duration,
            }],
            "running_losses": []
        }

        n_mini_batch: int = int((len(self.train_loader.dataset) / self.batch_size) / 5.0)

        for epoch in range(epochs):
            self.model.train()  # enter training mode

            batch_iterations = 0
            epoch_loss = 0.0
            running_loss = 0.0

            start = time.time()
            for image_set, label_set in self.train_loader:
                self.optimizer.zero_grad()  # don't forget this line!
                label_set = label_set.to(self.device)
                image_set = image_set.to(self.device)
                output = self.model(image_set)
                loss = self.loss_function(output, label_set)
                loss.backward()  # compute the derivatives of the model
                self.optimizer.step()  # update weights according to the optimizer

                batch_iterations += 1
                running_loss += loss.item()
                epoch_loss += loss.item()

                if batch_iterations % n_mini_batch == n_mini_batch - 1:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, batch_iterations + 1, running_loss / n_mini_batch))
                    summary["running_losses"].append(running_loss)
                    running_loss = 0.0

            training_duration_sec = int(time.time() - start)

            avg_epoch_loss = epoch_loss / batch_iterations

            accuracy, eval_duration = self.evaluate()
            summary['training'].append({
                "accuracy": accuracy,
                "train_duration": training_duration_sec,
                "test_duration": eval_duration,
                "avg_loss": avg_epoch_loss,
                "i": batch_iterations
            })

            print('Epoch {}:\tAccuracy {:07.5f}, Avg Loss {:07.5f} trained in {:d}s, evaluated in {:d}s'.format(
                epoch + 1,
                accuracy,
                avg_epoch_loss,
                training_duration_sec,
                eval_duration))

            # self.model.plot_layer(self.model.feature_extractor[0], 'conv1.png')

        filename = 'out_{}.json'.format(time.strftime("%Y%m%d-%H%M%S"))
        with open(filename, 'w') as fp:
            json.dump(summary, fp, indent=4)

    def evaluate(self) -> (float, float):

        start = time.time()

        self.model.eval()  # enter evaluation mode
        correct_label_count = 0

        for images, labels in self.validation_loader:
            labels = labels.to(self.device)
            images = images.to(self.device)
            output = self.softmax(self.model(images))
            predicted = torch.max(output, dim=1)[1]
            correct_label_count += (predicted == labels).sum().item()

        duration = int(time.time() - start)

        return correct_label_count / len(self.validation_loader.dataset), duration
