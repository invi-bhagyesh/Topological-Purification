import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetPP(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, features=[64, 128, 256]):
        super().__init__()
        self.features = features
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleDict()
        
        # Encoder
        for i, f in enumerate(self.features):
            input_ch = in_channels if i == 0 else self.features[i-1]
            self.encoder.append(ConvBlock(input_ch, f))
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder with corrected channel calculations
        for depth in range(1, len(self.features)):
            for step in range(len(self.features) - depth):
                in_ch = depth * self.features[step] + self.features[step+1]
                self.decoder[f'x{step}_{depth}'] = ConvBlock(in_ch, self.features[step])
        
        # Final convolution
        self.final = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc_features = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            enc_features.append(x)
            if i != len(self.encoder)-1:
                x = self.pool(x)
        
        # Initialize nested dictionary
        nest = {f'x{i}_0': feat for i, feat in enumerate(enc_features)}
        
        # Decoder path with dense connections
        for depth in range(1, len(self.features)):
            for step in range(len(self.features) - depth):
                key = f'x{step}_{depth}'
                prev_features = [nest[f'x{step}_{k}'] for k in range(depth)]
                up_feature = F.interpolate(nest[f'x{step+1}_{depth-1}'], 
                                         scale_factor=2, mode='bilinear', 
                                         align_corners=True)
                concat_features = torch.cat(prev_features + [up_feature], dim=1)
                nest[key] = self.decoder[key](concat_features)
        
        return self.final(nest[f'x0_{len(self.features)-1}'])
