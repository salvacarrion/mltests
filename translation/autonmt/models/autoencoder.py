
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, enc_dim):
        super().__init__()
        self.mode = "encode"
        self.encoder = nn.Sequential(nn.Linear(input_dim, enc_dim), nn.Tanh(), nn.Linear(enc_dim, enc_dim))
        self.decoder = nn.Sequential(nn.Linear(enc_dim, enc_dim), nn.Tanh(), nn.Linear(enc_dim, input_dim))

    def forward(self, x):
        if self.mode == "encode":
            embedding = self.encoder(x[0])
        else:
            embedding = self.decoder(x[0])
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch

        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
