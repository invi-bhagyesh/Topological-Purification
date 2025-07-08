#Victim Model

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
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


#Attacks

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    ori_images = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach_()
    return images

#Attack trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=9).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
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
    print(f"Epoch {epoch+1}/5, Loss: {running_loss/len(train_loader):.4f}")

#Combined Data(Clean + perturbed images)

epsilon = 0.1
alpha = 0.01
pgd_iters = 7

clean_images = []
clean_labels = []
adversarial_images = []
adversarial_labels = []

model.eval()

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    batch_size = images.size(0)
    half = batch_size // 2

    # FGSM on first half
    images_fgsm = images[:half].clone().detach()
    labels_fgsm = labels[:half].squeeze()
    images_fgsm.requires_grad = True
    outputs = model(images_fgsm)
    loss = F.cross_entropy(outputs, labels_fgsm)
    model.zero_grad()
    loss.backward()
    data_grad = images_fgsm.grad.data
    fgsm_images = fgsm_attack(images_fgsm, epsilon, data_grad)

    # PGD on second half
    images_pgd = images[half:].clone().detach()
    labels_pgd = labels[half:].squeeze()
    pgd_images = pgd_attack(model, images_pgd, labels_pgd, epsilon, alpha, pgd_iters)

    # Collect clean and adversarial images and labels
    clean_images.append(images)
    clean_labels.append(torch.zeros(batch_size, dtype=torch.long, device=device))  # 0 for clean

    adv_batch = torch.cat([fgsm_images, pgd_images], dim=0)
    adversarial_images.append(adv_batch)
    adversarial_labels.append(torch.ones(batch_size, dtype=torch.long, device=device))  # 1 for perturbed

clean_images = torch.cat(clean_images, dim=0)
clean_labels = torch.cat(clean_labels, dim=0)
adversarial_images = torch.cat(adversarial_images, dim=0)
adversarial_labels = torch.cat(adversarial_labels, dim=0)

all_images = torch.cat([clean_images, adversarial_images], dim=0)
all_labels = torch.cat([clean_labels, adversarial_labels], dim=0)

combined_dataset = TensorDataset(all_images, all_labels)
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
