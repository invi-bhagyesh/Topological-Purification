import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleDict()

        # Encoder blocks
        for i, f in enumerate(features):
            prev_channels = in_channels if i == 0 else features[i - 1]
            self.encoder.append(ConvBlock(prev_channels, f))

        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder blocks with nested connections
        for depth in range(1, len(features)):
            for step in range(len(features) - depth):
                key = f"x{step}_{depth}"
                in_ch = features[step + 1] * depth + features[step]
                self.decoder[key] = ConvBlock(in_ch, features[step])

        # Final conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc = []
        for block in self.encoder:
            x = block(x)
            enc.append(x)
            x = self.pool(x)

        nest = dict()
        for i in range(len(enc)):
            nest[f"x{i}_0"] = enc[i]

        for depth in range(1, len(enc)):
            for step in range(len(enc) - depth):
                prev = [nest[f"x{step}_{k}"] for k in range(depth)]
                prev.append(self.upsample(nest[f"x{step+1}_{depth-1}"]))
                nest[f"x{step}_{depth}"] = self.decoder[f"x{step}_{depth}"](torch.cat(prev, dim=1))

        out = self.final(nest["x0_3"])  # final deep nested output
        return out

"""
model = UNetPP(in_channels=1, out_channels=2)
x = torch.randn(1, 1, 128, 128)
out = model(x)                               # Output: [1, 2, 128, 128]

# Get prediction
probs = torch.softmax(out, dim=1)
pred_mask = torch.argmax(probs, dim=1)       # [1, 128, 128]

"""