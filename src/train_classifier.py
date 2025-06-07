import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import wandb
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataloader import MNIST  
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

        val_acc, val_loss = evaluate(model, val_loader, device)
        train_acc = correct/total
        train_loss = running_loss/len(train_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # log to wandb
        wandb.log({
        "epoch": epoch + 1,
        "loss/train": train_loss,
        "acc/train": train_acc,
        "loss/val": val_loss,
        "acc/val": val_acc
        })

   
    if file_name is not None:
        torch.save(model.state_dict(), file_name)

    return model

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss/len(dataloader)
    accuracy = correct/total
    return accuracy, avg_val_loss

if not os.path.isdir('models'):
    os.makedirs('models')

if __name__== "__main__":
    # Call training function
    train(MNIST(), "models/example_classifier.pth", [32, 32, 64, 64, 200, 200], num_epochs=50)
