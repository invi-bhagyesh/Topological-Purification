autoencoder = SimpleAutoencoder()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder.load_state_dict(torch.load('/kaggle/input/required8/pathmnist_autoencoder.pth', map_location=device))
autoencoder = autoencoder.to(device)
autoencoder.eval()

classifier = PerturbationClassifier().to(device)
clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
clf_criterion = nn.CrossEntropyLoss()
EPOCHS = 20

f1_history = []
precision_history = []
recall_history = []
accuracy_history = []
loss_history = []

for epoch in range(EPOCHS):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for images, labels in combined_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            reconstructed = autoencoder(images)
        
        clf_optimizer.zero_grad()
        outputs = classifier(reconstructed)
        loss = clf_criterion(outputs, labels)
        loss.backward()
        clf_optimizer.step()
        
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
    loss_history.append(running_loss / len(combined_loader))
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(combined_loader):.4f}, "
          f"Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Save the trained classifier
torch.save(classifier.state_dict(), "perturbation_classifier.pth")
print("Classifier model saved to perturbation_classifier.pth")
