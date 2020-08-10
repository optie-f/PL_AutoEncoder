import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, KMNIST, FashionMNIST, CIFAR10


dataset_classes = {
    'MNIST': MNIST,
    'KMNIST': KMNIST,
    'FashionMNIST': FashionMNIST,
    'CIFAR10': CIFAR10
}


class NormalizeAllChannel():
    """
    a normalization transform using given mean & std
    but accepts any number of channels.
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
            self.Dataset = dataset_classes[data_name]
        else:
            raise NotImplementedError

    def prepare_data(self):
        self.Dataset(self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset = self.Dataset(
                self.data_dir,
                train=True,
                transform=self.transform
            )
            size = len(self.dataset)
            t, v = (int(size * 0.9), int(size * 0.1))
            t += (t + v != size)
            self.dataset_train, self.dataset_val = random_split(self.dataset, [t, v])

        if stage == 'test' or stage is None:
            self.dataset_test = self.Dataset(
                self.data_dir,
                train=False,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
        )
