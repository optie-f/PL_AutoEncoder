from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from modules.model import AutoEncoder
from modules.data_loader import DataModule
from argparse import ArgumentParser


def train(data_name, hparams):
    train_loader = DataModule(data_name=data_name)
    train_loader.prepare_data()
    train_loader.setup()

    in_dim = np.prod(train_loader.dataset[0][0].size())
    dimentions = [in_dim, 512, 128, 64, 12, 2]

    autoEncoder = AutoEncoder(dimentions)
    print(autoEncoder)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix=''
    )

    logger = TensorBoardLogger('log', name=data_name)
    trainer = Trainer(
        logger=logger,
        default_root_dir='./log',
        checkpoint_callback=checkpoint_callback,
        row_log_interval=50,
        max_epochs=hparams.max_epochs,
        gpus=hparams.gpus,
        tpu_cores=hparams.tpu_cores
    )

    trainer.fit(autoEncoder, train_loader)


def main():
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--tpu_cores', default=None)
    parser.add_argument('--max_epochs', default=50)
    args = parser.parse_args()

    for data_name in ['MNIST', 'FashionMNIST', 'KMNIST', 'CIFAR10']:
        train(data_name, args)


if __name__ == "__main__":
    main()
