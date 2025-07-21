#python MEDMNIST/model/Reformer/Classifier/train.py --epochs 30 --batch_size 32 --lr 1e-3 --device cuda --data_flag pathmnist
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os

# Go up 3 levels to reach the root (where set_root_path.py is)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from set_root_path import add_project_root
add_project_root()

from MEDMNIST.dataloader import load_medmnist
from UNet import UNet

def train_unet_classifier(train_loader, val_loader, device, input_channels, num_classes, epochs=30, lr=1e-3):
    model = UNet(input_channels=input_channels, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_model = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.squeeze().to(device)
                val_outputs = model(val_images)
                _, val_predicted = torch.max(val_outputs, 1)
                val_correct += (val_predicted == val_labels).sum().item()
                val_total += val_labels.size(0)
        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()

    if best_model is not None:
        torch.save(best_model, 'unet_classifier.pth')
        print("Best UNet classifier saved to unet_classifier.pth")
    return model

def main():
    parser = argparse.ArgumentParser(description='Train UNet classifier for MedMNIST')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--data_flag', type=str, default='pathmnist', help='MedMNIST dataset flag (e.g., pathmnist, bloodmnist, etc.)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _, info = load_medmnist(data_flag=args.data_flag, batch_size=args.batch_size)
    input_channels = info.get('n_channels', 3)
    num_classes = len(info['label'])
    train_unet_classifier(train_loader, val_loader, device, input_channels, num_classes, epochs=args.epochs, lr=args.lr)

if __name__ == '__main__':
    main()

