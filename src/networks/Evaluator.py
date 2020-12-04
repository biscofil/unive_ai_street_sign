import torch.nn as nn


class Evaluator(object):

    def __init__(self, model: nn.Module, device):
        self.model = model.to(device)
        self.device = device
        self.softmax = nn.Softmax(dim=1)

    def evaluate_images(self, images):
        images = images.to(self.device)
        return self.softmax(self.model(images))
