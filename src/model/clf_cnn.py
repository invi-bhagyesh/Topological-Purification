import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, params):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, params[0], kernel_size=3)   
        self.conv2 = nn.Conv2d(params[0], params[1], kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(params[1], params[2], kernel_size=3)
        self.conv4 = nn.Conv2d(params[2], params[3], kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)

       
        self.flatten_dim = self._get_flatten_dim()

        self.fc1 = nn.Linear(self.flatten_dim, params[4])
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(params[4], params[5])
        self.fc3 = nn.Linear(params[5], 10)

    def _get_flatten_dim(self):
        dummy_input = torch.zeros(1, 1, 28, 28)  
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        return x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
