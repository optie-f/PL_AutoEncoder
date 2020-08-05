from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from modules.model import AutoEncoder

dimentions = [784, 128, 64, 12, 2]

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5),
        (0.5)
    )  # [0,1] => [-1,1]
])

train_dataset = MNIST('./data', download=True, transform=img_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

autoEncoder = AutoEncoder(dimentions)
print(autoEncoder)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    verbose=True,
    monitor='loss',
    mode='min',
    prefix=''
)

logger = TensorBoardLogger('log', name='mnist')
trainer = Trainer(
    logger=logger,
    default_root_dir='./checkpoints', checkpoint_callback=checkpoint_callback
)

trainer.fit(autoEncoder, train_loader)
