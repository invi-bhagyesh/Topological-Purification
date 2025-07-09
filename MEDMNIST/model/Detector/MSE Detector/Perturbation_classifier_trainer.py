import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from Simple_Autoencoder import SimpleAutoencoder
from Perturbation_classifier import PerturbationClassifier

def train_classifier(autoencoder, train_loader, device, epochs=20, lr=0.001):
    """
    Train the classifier using reconstructed images from the autoencoder
    """
    classifier = PerturbationClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    f1_history = []
    precision_history = []
    recall_history = []
    accuracy_history = []
    loss_history = []
    
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get reconstructed images from pre-trained autoencoder
            with torch.no_grad():
                reconstructed = autoencoder(images)
            
            # Classify reconstructed images
            outputs = classifier(reconstructed)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        accuracy_history.append(accuracy)
        loss_history.append(running_loss / len(train_loader))
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    torch.save(classifier.state_dict(), "mse_classifier.pth")
    print("Classifier saved to mse_classifier.pth")
    return classifier

if __name__ == '__main__':
    # For standalone execution
    from MEDMNIST.dataloader import load_medmnist
    from MEDMNIST.Attack_generation import SimpleCNN, train_model, generate_mixed_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained autoencoder
    autoencoder = SimpleAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load('mse_autoencoder.pth', map_location=device))
    autoencoder.eval()
    
    # Get mixed data loader
    train_loader, _, _, _ = load_medmnist(batch_size=32)
    attack_classifier = SimpleCNN(num_classes=9).to(device)
    mixed_loader = generate_mixed_dataset(attack_classifier, train_loader, device=device)
    
    train_classifier(autoencoder, mixed_loader, device)
