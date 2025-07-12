import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*4*4, latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=512, output_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        decoded = self.decoder(z)
        # Center crop to 28x28 if the output is larger
        if decoded.shape[-1] > 28 or decoded.shape[-2] > 28:
            return TF.center_crop(decoded, [28, 28])
        return decoded


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
