import sys
sys.path.append('/kaggle/input/required5')
from src.dataloader import MNIST
from src.model.DAE_model import DenoisingAutoEncoder as DAE
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 100
reg_strength = 1e-9
activation = "sigmoid"

# Architectures
shape = (1, 28, 28)
combination_I = [3, "average", 3]
combination_II = [3]

# Load data using your class
data = MNIST(batch_size=batch_size)

def train_autoencoder(model, train_loader, val_loader, archive_name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=reg_strength)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, _ in train_loader:
            # Add Gaussian noise
            noisy_imgs = imgs + 0.1 * torch.randn_like(imgs)
            noisy_imgs = torch.clamp(noisy_imgs, -1.0, 1.0)

            noisy_imgs, imgs = noisy_imgs.to(device), imgs.to(device)

            # Forward + backward
            outputs = model(noisy_imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("./defensive_models/", exist_ok=True)
    torch.save(model.state_dict(), f"./defensive_models/{archive_name}.pth")
    print(f"Model saved as ./defensive_models/{archive_name}.pth")

# Train Model I
AE_I = DAE(shape, combination_I, v_noise=0.1, activation=activation, reg_strength=reg_strength)
train_autoencoder(AE_I, data.train_loader, data.validation_loader, "MNIST_I")

# Train Model II
AE_II = DAE(shape, combination_II, v_noise=0.1, activation=activation, reg_strength=reg_strength)
train_autoencoder(AE_II, data.train_loader, data.validation_loader, "MNIST_II")
