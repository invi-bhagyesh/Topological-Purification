import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=9, features=[32, 64, 128]):
        super().__init__()
        self.enc1 = UNetBlock(input_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = UNetBlock(features[2], features[2])

        self.up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec3 = UNetBlock(features[2], features[1])
        self.up2 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec2 = UNetBlock(features[1] + features[0], features[0])
        self.up1 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.dec1 = UNetBlock(features[0] + input_channels, features[0])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[0], num_classes)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        # For the last skip connection, upsampled feature and input
        d1 = self.dec1(torch.cat([u1, x], dim=1))

        # Classification head
        logits = self.classifier(d1)
        return logits
