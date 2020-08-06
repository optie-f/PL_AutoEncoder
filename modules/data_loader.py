import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10


dataset_classes = {
    'MNIST': MNIST,
    'KMNIST': KMNIST,
    'FashionMNIST': FashionMNIST,
    'CIFAR10': CIFAR10
}


class NormalizeAllChannel():
    """
    By default, [0, 1] -> [-1, 1]
    """

    def __init__(self, mean=0.5, std=0.5):
        self.mean = 0.5
        self.std = 0.5

    def __call__(self, x):
        return (x - self.mean) / self.std


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', data_name='', batchsize=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batchsize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeAllChannel()
        ])
        if data_name in dataset_classes:
            self.dataset_class = dataset_classes[data_name]
        else:
            raise NotImplementedError

    def prepare_data(self):
        self.dataset_class(self.data_dir, download=True)

    def setup(self):
        self.dataset = self.dataset_class(
            self.data_dir,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
