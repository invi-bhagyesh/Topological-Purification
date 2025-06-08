import torch
import torch.nn as nn
import torch.nn.functional as F  
from torch.autograd import Function

class AutoEncoder(nn.Module):
    def __init__(self, input_shape=(4, 160, 224, 160)):
        super(AutoEncoder, self).__init__()
        self.input_shape = input_shape
        
        # Revised Encoder with valid pooling parameters
        self.conv1 = nn.Conv3d(4, 16, 3, padding=1)
        self.pool1 = nn.MaxPool3d(2, stride=2, return_indices=True)  # 160→80
        
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2, stride=2, return_indices=True)  # 80→40
        
        self.conv3 = nn.Conv3d(32, 96, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, stride=2, return_indices=True)  # 40→20

        # Dynamic size calculation
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.pool1(F.relu(self.conv1(dummy)))[0]
            dummy = self.pool2(F.relu(self.conv2(dummy)))[0]
            dummy = self.pool3(F.relu(self.conv3(dummy)))[0]
            self.flatten_size = dummy.numel()  # Now calculates 96*20*56*20 = 215,040

        # Adjusted linear layers
        self.enc_linear = nn.Linear(self.flatten_size, 512)
        self.dec_linear = nn.Linear(512, self.flatten_size)

        # Decoder with proper transpose convs
        self.deconv3 = nn.ConvTranspose3d(96, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 16, 3, padding=1)
        self.deconv1 = nn.ConvTranspose3d(16, 4, 3, padding=1)
        
    def encode(self, x):
        # Encoder with valid dimension progression
        x, idx1 = self.pool1(F.relu(self.conv1(x)))  # 160→80
        x, idx2 = self.pool2(F.relu(self.conv2(x)))  # 80→40
        x, idx3 = self.pool3(F.relu(self.conv3(x)))  # 40→20
        
        orig_shape = x.shape
        x = x.view(x.size(0), -1)
        x = self.enc_linear(x)
        return x, (idx1, idx2, idx3, orig_shape)

    def decode(self, x, pool_data):
        idx1, idx2, idx3, orig_shape = pool_data
        x = self.dec_linear(x).view(orig_shape)
        
        # Proper unpooling sequence
        x = F.max_unpool3d(x, idx3, 2, 2)  # 20→40
        x = F.relu(self.deconv3(x))
        
        x = F.max_unpool3d(x, idx2, 2, 2)  # 40→80
        x = F.relu(self.deconv2(x))
        
        x = F.max_unpool3d(x, idx1, 2, 2)  # 80→160
        x = torch.sigmoid(self.deconv1(x))
        
        return x

    def forward(self, x):
        encoded, pool_data = self.encode(x)
        return self.decode(encoded, pool_data)
