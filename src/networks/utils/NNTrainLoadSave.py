from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.utils import save_image


class NNTrainLoadSave(nn.Module):

    def __init__(self, output_filename: str):
        super().__init__()
        self.output_filename = output_filename

    def load_from_file(self) -> bool:
        my_file = Path(self.output_filename)
        if my_file.is_file():
            # file exists
            try:
                print("Loading model from {} ...".format(self.output_filename))
                self.load_state_dict(torch.load(self.output_filename))
                self.eval()
                self.to(self.device_name)
                print("[DONE]")
                return True
            except:
                print("Loading failed! Did the structure change?")
        return False

    def store_to_file(self) -> None:
        print("Storing model to {} ...".format(self.output_filename))
        torch.save(self.state_dict(), self.output_filename)
        print("[DONE]")

    @staticmethod
    def plot_layer(layer, filename: str) -> None:
        import matplotlib.pyplot as plt
        import math
        kernels = layer.weight.detach().clone().cpu()
        # check size for sanity check
        # print(kernels.size())
        # normalize to (0,1) range so that matplotlib can plot them
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()

        n_filters = int(kernels.size()[0])
        rows: int = math.ceil(math.sqrt(n_filters))

        filter_img = torchvision.utils.make_grid(kernels, nrow=rows)
        # change ordering since matplotlib requires images to be (H, W, C)
        plt.imshow(filter_img.permute(1, 2, 0))
        # You can directly save the image as well using
        save_image(kernels, filename, nrow=rows)
