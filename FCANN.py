import torch.nn as nn


class FCANN(nn.Module):

    def __init__(self, img_size: int):
        super().__init__()
        self.lin1 = nn.Linear(img_size * img_size * 3, 1024)  # TODO change input shape
        self.bn1 = nn.BatchNorm1d(1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.lin3 = nn.Linear(1024, 43)
        self.bn = nn.BatchNorm1d(43)
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, -1)
        x = self.relu(self.lin1(x))
        x = self.bn1(x)
        x = self.relu(self.lin2(x))
        x = self.bn2(x)
        x = self.lin3(x)
        return x
