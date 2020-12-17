import json
import time

import matplotlib.pyplot as plt
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

    def train(self, max_epochs: int, max_patience: int, ui: bool = False) -> None:

        print("\tEpochs :", max_epochs)
        print("\tLearning Rate :", self.learning_rate)
        print("\tMomentum :", self.momentum)
        print("\tBatch Size :", self.batch_size)
        print("\tWeight Decay :", self.weight_decay)

        test_accuracy, test_loss, test_duration = self.evaluate()

        print('Initial test accuracy: {:07.5f}. Loss: {:07.5f}'.format(test_accuracy, test_loss))
        summary = {
            "structure": self.model.__str__(),
            "patience": max_patience,
            "epochs": max_epochs,
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "training": [{
                "test_accuracy": test_accuracy,
                "test_duration": test_duration,
                "test_loss": test_loss
            }],
            "running_losses": []
        }

        if ui:
            plt.ion()
            ax = plt.axes()
            ax.set_ylim([0, 1])
            ax.plot([d["test_loss"] for d in summary["training"]])
            plt.draw()
            plt.pause(0.1)

        n = len(self.train_loader.dataset)

        n_mini_batch: int = int((n / self.batch_size) / 5.0)

        min_test_loss = test_loss

        patience = max_patience
        state_dict = self.model.state_dict()

        for epoch in range(max_epochs):
            self.model.train()  # enter training mode

            batch_iterations = 0
            epoch_loss = 0.0
            epoch_partition_loss = 0.0
            epoch_partition_count = 0
            correct_label_count = 0

            start = time.time()
            for image_set, label_set in self.train_loader:
                epoch_partition_count += 1
                self.optimizer.zero_grad()  # don't forget this line!
                label_set = label_set.to(self.device)
                image_set = image_set.to(self.device)
                output = self.model(image_set)
                predicted = torch.max(self.softmax(output), dim=1)[1]
                loss = self.loss_function(output, label_set)
                loss.backward()  # compute the derivatives of the model
                self.optimizer.step()  # update weights according to the optimizer
                correct_label_count += (predicted == label_set).sum().item()
                batch_iterations += 1
                loss_v = loss.item()
                epoch_partition_loss += loss_v
                epoch_loss += loss_v

                if batch_iterations % n_mini_batch == n_mini_batch - 1:  # print every [n_mini_batch] mini-batches
                    print('[%d, %5d] training loss: %.5f' %
                          (epoch + 1, batch_iterations + 1, epoch_partition_loss / epoch_partition_count))
                    summary["running_losses"].append(epoch_partition_loss / epoch_partition_count)
                    epoch_partition_loss = 0.0
                    epoch_partition_count = 0

            training_duration = int(time.time() - start)

            train_accuracy = correct_label_count / n
            train_loss = epoch_loss / batch_iterations

            test_accuracy, test_loss, test_duration = self.evaluate()
            summary['training'].append({
                "train_accuracy": train_accuracy,
                "train_duration": training_duration,
                "train_loss": train_loss,
                "test_accuracy": test_accuracy,
                "test_duration": test_duration,
                "test_loss": test_loss,
                "i": batch_iterations,
                "patience": patience
            })

            print('E {}:\tTR.A {:07.5f}, TE.A {:07.5f}, TR.L {:07.5f}, TE.L {:07.5f}, TR.T: {:d}s, TE.T: {:d}s'.format(
                epoch + 1,
                train_accuracy,
                test_accuracy,
                train_loss,
                test_loss,
                training_duration,
                test_duration))

            if test_loss < min_test_loss:
                # best so far
                print("best so far, store state_dict!")
                min_test_loss = test_loss
                state_dict = self.model.state_dict()
                patience = max_patience
            else:
                # not the best so far, check patience
                patience -= 1
                if patience == 0:
                    print("patience = 0")
                    break  # quit

            if ui:
                ax.plot([d["test_loss"] for d in summary["training"]])
                plt.draw()
                plt.pause(0.1)

        # restore best
        self.model.load_state_dict(state_dict)
        # self.model.plot_layer(self.model.feature_extractor[0], 'conv1.png')

        filename = 'out_{}.json'.format(time.strftime("%Y%m%d-%H%M%S"))
        with open(filename, 'w') as fp:
            json.dump(summary, fp, indent=4)

    def evaluate(self) -> (float, float, float):

        start = time.time()

        self.model.eval()  # enter evaluation mode
        correct_label_count = 0
        eval_loss = 0
        i = 0

        for images, labels in self.validation_loader:
            labels = labels.to(self.device)
            images = images.to(self.device)
            output = self.model(images)
            loss = self.loss_function(output, labels)
            output_sm = self.softmax(output)
            eval_loss += loss.item()
            predicted = torch.max(output_sm, dim=1)[1]
            correct_label_count += (predicted == labels).sum().item()
            i += 1

        duration = int(time.time() - start)

        return correct_label_count / len(self.validation_loader.dataset), eval_loss / i, duration
