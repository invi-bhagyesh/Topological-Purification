import sys
sys.path.append('/kaggle/input/required4')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.dataloader import MNIST  
from model.model import CNNClassifier  

def train(data, file_name, params, num_epochs=50, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    model = CNNClassifier(params).to(device)


    train_loader = data.train_loader
    val_loader = data.validation_loader

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)

   
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in data.train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Acc: {correct/total:.4f}, Val Acc: {val_acc:.4f}")

   
    if file_name is not None:
        torch.save(model.state_dict(), file_name)

    return model

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

if not os.path.isdir('models'):
    os.makedirs('models')

# Call training function
train(MNIST(), "models/example_classifier.pth", [32, 32, 64, 64, 200, 200], num_epochs=50)
