import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from FCANN import FCANN
from StreetSignDataset import StreetSignDataset
from Trainer import Trainer
from custom_transformations.CustomCrop import CustomCrop
from custom_transformations.CustomRescale import CustomRescale
from custom_transformations.CustomToTensor import CustomToTensor

if __name__ == '__main__':

    # fig = plt.figure()
    # for i in range(len(face_dataset)):
    #     sample = face_dataset[i]
    #     print(i, sample['image']['Filename'])
    #     im = sample['image_data']
    #     plt.imshow(im)
    #     plt.show()
    #     plt.clf()  # will make the plot window empty
    #     im.close()

    config = {
        'lr': 1e-4,
        'momentum': 0.9,
        'weight_decay': 0.001,
        'batch_size': 8,
        'epochs': 10,
        'device': 'cpu',  # 'cuda:0',
        'seed': 314
    }

    # set the seeds to repeat the experiments
    if 'cuda' in config['device']:
        torch.cuda.manual_seed_all(config['seed'])
    else:
        torch.manual_seed(config['seed'])

    img_size = 35

    transform = transforms.Compose([
        # {PIL, rect}
        CustomCrop(),
        # PIL
        CustomRescale(img_size),
        # PIL
        CustomToTensor(),
        # Tensor
    ])
    ssd = StreetSignDataset(transform=transform)

    image_count = len(ssd)
    train_img_count = int(image_count * 0.8)
    train_dataset, test_dataset = torch.utils.data.random_split(ssd, [train_img_count, image_count - train_img_count])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = FCANN(img_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0.001)

    trainer = Trainer(model, config['device'], train_loader, test_loader, criterion, optimizer)
    trainer.train(epochs=10)
