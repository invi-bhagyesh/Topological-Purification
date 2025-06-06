import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class MNIST:
    def __init__(self, batch_size=128):
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Automatically downloads if not available
        full_train = datasets.MNIST(root='/kaggle/working', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(root='/kaggle/working', train=False, download=True, transform=transform)

        # Split training and validation sets
        val_size = 5000
        train_size = len(full_train) - val_size
        self.train_data, self.validation_data = random_split(full_train, [train_size, val_size])

        # DataLoaders
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_data, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size)

    @staticmethod
    def print():
        return "MNIST dataset loaded."
