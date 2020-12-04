import torch
from PIL import Image

from src.FrameAnalyzer import FrameAnalyzer
from src.dataset.GTSRBStreetSignDataset import GTSRBStreetSignDataset
from src.networks.FullNetwork import FullNetwork

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 852144

if 'cuda' in device:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

model = FullNetwork(device)


def train():
    model.train_model()


def realtime_detection():
    fa = FrameAnalyzer(model, device)
    # fa.video_analyzer()

    images = [
        {'path': '40.jpeg', 'exp': '40kmh'},
        {'path': 'no.jpeg', 'exp': 'no something'},
        {'path': 'stop1.png', 'exp': 'stop'},
        {'path': 'ahead.jpg', 'exp': 'ahead'},
        {'path': 'priority.jpg', 'exp': 'yellow'},
    ]

    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))

    for image in images:
        print("Expected :", image['exp'])
        image_filename = image['path']
        im = Image.open(dir_path + '/dataset/new/' + image_filename)
        label_idx = fa.get_patch_label(im)
        print("\tActual :", GTSRBStreetSignDataset.get_label_names()[label_idx])


if __name__ == '__main__':
    train()
    #realtime_detection()
