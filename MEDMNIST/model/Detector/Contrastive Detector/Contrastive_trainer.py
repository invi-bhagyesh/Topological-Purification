import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class ContrastiveTrainer:
    def __init__(self, device, batch_size=32, temperature=0.1, n_views=2):
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_views = n_views
        
    def info_nce_loss(self, features):
        """
        InfoNCE loss for contrastive learning
        Args:
            features: [n_views * batch_size, embedding_dim]
        """
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

def train_contrastive_encoder(train_loader, val_loader, device, epochs=30, lr=1e-3):
    """
    Train the contrastive encoder using InfoNCE loss
    """
    model = ContrastiveEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    trainer = ContrastiveTrainer(device, batch_size=32, temperature=0.1, n_views=2)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Create two views of the same images (you can add augmentations here)
            view1 = images
            view2 = images  # In practice, you'd apply different augmentations
            
            # Concatenate views
            combined_images = torch.cat([view1, view2], dim=0)
            
            # Get embeddings
            features = model(combined_images)
            
            # Compute InfoNCE loss
            logits, labels = trainer.info_nce_loss(features)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                view1 = val_images
                view2 = val_images
                combined_val = torch.cat([view1, view2], dim=0)
                val_features = model(combined_val)
                val_logits, val_labels = trainer.info_nce_loss(val_features)
                batch_loss = nn.CrossEntropyLoss()(val_logits, val_labels)
                val_loss += batch_loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
    
    torch.save(model.state_dict(), 'contrastive_encoder.pth')
    print("Contrastive encoder saved to contrastive_encoder.pth")
    return model

def train_contrastive_classifier(encoder, train_loader, device, epochs=20, lr=0.001):
    """
    Train the classifier using embeddings from the contrastive encoder
    """
    classifier = ContrastiveClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    f1_history = []
    precision_history = []
    recall_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get embeddings from pre-trained encoder
            with torch.no_grad():
                embeddings = encoder(images)
            
            # Classify embeddings
            outputs = classifier(embeddings)
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
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
              f"Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    torch.save(classifier.state_dict(), "contrastive_classifier.pth")
    print("Contrastive classifier saved to contrastive_classifier.pth")
    return classifier 