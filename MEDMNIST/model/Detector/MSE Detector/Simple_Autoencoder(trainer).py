import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

EPOCHS = 30
USE_DENOISING = False      # Set True for denoising autoencoder
LOSS_TYPE = 'mse'          # 'mse' or 'l1'
LEARNING_RATE = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleAutoencoder().to(device)
criterion = nn.MSELoss() if LOSS_TYPE == 'mse' else nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)
        if USE_DENOISING:
            noisy_images = images + 0.2 * torch.randn_like(images)
            noisy_images = torch.clamp(noisy_images, 0., 1.)
            inputs = noisy_images
        else:
            inputs = images

        outputs = model(inputs)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx}], Train Loss: {loss.item():.4f}")

    # ---- VALIDATION (once per epoch) ----
    model.eval()
    val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for val_images, _ in val_loader:
            val_images = val_images.to(device)
            if USE_DENOISING:
                noisy_val = val_images + 0.2 * torch.randn_like(val_images)
                noisy_val = torch.clamp(noisy_val, 0., 1.)
                val_inputs = noisy_val
            else:
                val_inputs = val_images
            val_outputs = model(val_inputs)
            batch_loss = criterion(val_outputs, val_images)
            val_loss += batch_loss.item()
            num_batches += 1
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / num_batches
    print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)

torch.save(model.state_dict(), 'pathmnist_autoencoder.pth')
print("Model weights saved to pathmnist_autoencoder.pth")
