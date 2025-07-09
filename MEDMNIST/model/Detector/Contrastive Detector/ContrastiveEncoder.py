import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_channels=3, projection_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*4*4, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        features = self.projection(z)
        return F.normalize(features, dim=1)