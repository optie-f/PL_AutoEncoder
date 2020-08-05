from torch import nn
from torch import optim
from torchvision.utils import make_grid
import pytorch_lightning as pl
from math import sqrt


class LAN(nn.Module):
    """
    Linear-Activation-Normalization.
    """

    def __init__(self, in_dim, out_dim, activation=nn.ReLU(True), norm=nn.Identity()):
        super(LAN, self).__init__()
        self.L = nn.Linear(in_dim, out_dim)
        self.A = activation
        self.N = norm

    def forward(self, x):
        z = self.L(x)
        z = self.A(z)
        z = self.N(z)
        return z


class DENcoder(nn.Module):
    def __init__(self, dimentions, mid_activation, last_activation):
        super(DENcoder, self).__init__()
        layers = []
        in_dims = dimentions[:-1]

        for i, in_dim in enumerate(in_dims):
            activation = last_activation if i + 1 == len(in_dims) else mid_activation
            out_dim = dimentions[i + 1]
            layers.append(
                LAN(in_dim, out_dim, activation=activation)
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, in_dimentions):
        super(AutoEncoder, self).__init__()
        out_dimentions = list(reversed(in_dimentions))
        self.encoder = DENcoder(in_dimentions, nn.ReLU(True), nn.Identity())
        self.decoder = DENcoder(out_dimentions, nn.ReLU(True), nn.Tanh())
        self.criterion = nn.MSELoss()

    def forward(self, img):
        x = img.view(img.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        img_recon = x_hat.view(img.size())
        return img_recon

    def training_step(self, batch, batch_idx):
        img, _ = batch
        img_recon = self.forward(img)
        loss = self.criterion(img, img_recon)

        if self.global_step % self.trainer.row_log_interval == 0:
            sqrt_nb = int(sqrt(img.size(0)))
            self.logger.experiment.add_image(
                "image/original",
                make_grid(img, sqrt_nb, normalize=True, range=(-1, 1)),
                self.global_step
            )
            self.logger.experiment.add_image(
                "image/reconstructed",
                make_grid(img_recon, sqrt_nb, normalize=True, range=(-1, 1)),
                self.global_step
            )

        return {'loss': loss, 'log': {'loss': loss}}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
