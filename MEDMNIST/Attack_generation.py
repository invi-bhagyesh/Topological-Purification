import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchattacks  

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, device='cuda', epochs=5, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")


def generate_mixed_dataset(model, train_loader, epsilon=0.1, alpha=0.01, pgd_iters=7, device='cuda', pure_adv=False):
    clean_images, clean_labels = [], []
    adv_images, adv_labels = [], []

    # Initialize attacks
    fgsm = torchattacks.FGSM(model, eps=epsilon)
    pgd = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=pgd_iters)

    model.eval()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        half = batch_size // 2

        # FGSM
        fgsm_input = images[:half]
        fgsm_labels = labels[:half].squeeze()
        fgsm_output = fgsm(fgsm_input, fgsm_labels)

        # PGD
        pgd_input = images[half:]
        pgd_labels = labels[half:].squeeze()
        pgd_output = pgd(pgd_input, pgd_labels)

        combined_adv = torch.cat([fgsm_output, pgd_output], dim=0)
        adv_images.append(combined_adv)
        adv_labels.append(labels)

        if not pure_adv:
            clean_images.append(images)
            clean_labels.append(labels)

    # Stack and return
    if pure_adv:
        all_images = torch.cat(adv_images, dim=0)
        all_labels = torch.cat(adv_labels, dim=0)
    else:
        all_images = torch.cat(clean_images + adv_images, dim=0)
        all_labels = torch.cat(clean_labels + adv_labels, dim=0)

    combined_dataset = TensorDataset(all_images, all_labels)
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
    return combined_loader
