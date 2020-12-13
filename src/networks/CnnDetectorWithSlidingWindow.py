import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from src.networks.CnnDetector import CnnDetector


class CnnDetectorWithSlidingWindow(CnnDetector):

    def __init__(self, device_name: str, patch_size: int = 28, patch_stride: int = 10):
        super().__init__(device_name)
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x):

        x = x.unfold(1, self.patch_size, self.patch_stride) \
            .unfold(2, self.patch_size, self.patch_stride) \
            .unfold(3, self.patch_size, self.patch_stride)

        x = x.contiguous().view(-1, 3, self.patch_size, self.patch_size)

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

        x = torch.softmax(x, dim=0)

        # return labels, score
        return torch.argmax(x, dim=1), torch.max(x, dim=1)[0]

    def run_sliding_window(self, img: Image) -> (list, list):
        image_tensor: Tensor = (ToTensor())(img).float()
        image_tensor = image_tensor.to(self.device)
        labels, scores = self(image_tensor)
        return labels.tolist(), scores.tolist()
