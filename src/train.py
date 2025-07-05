import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 8
reg_strength = 1e-9
activation = "relu"

# --- Input shape for 2D slices: (C, H, W) ---
input_shape = (4, 240, 240)

# --- Architecture structures for 2D Conv DAE ---
# Very lightweight for 2D slice-based BraTS
structure_AE_I = [
    32,        # Conv2d: 4 → 16
    "max",     # → 120x120
    64,        # Conv2d
    "max",
    128,
    "max",
    "linear_bottleneck",
    2048
            # Bottleneck features
]



structure_AE_II = [
    64,
    "max",      # -> 120x120
    128,
    "max",      # -> 60x60
    256,
    "max",      # -> 30x30
    512        
]


def train_autoencoder(model, train_loader, val_loader, archive_name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=reg_strength)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print(f"\n--- Starting training for {archive_name} ---")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            imgs = batch_data['image']  # Expect shape: [B, 4, 240, 240]

            if epoch == 0 and batch_idx == 0:
                print(f"Input batch shape: {imgs.shape}")  # [B, 4, 240, 240]
                assert imgs.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {imgs.shape}"

            noisy_imgs = imgs + model.v_noise * torch.randn_like(imgs)
            noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)

            noisy_imgs, imgs = noisy_imgs.to(device), imgs.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                imgs = batch_data['image']
                noisy_imgs = imgs + model.v_noise * torch.randn_like(imgs)
                noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)
                noisy_imgs, imgs = noisy_imgs.to(device), imgs.to(device)
                outputs = model(noisy_imgs)
                loss = criterion(outputs, imgs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("./defensive_models/", exist_ok=True)
            torch.save(model.state_dict(), f"./defensive_models/{archive_name}_best.pth")
            print(f"  --> Best model for {archive_name} saved with Val Loss: {best_val_loss:.4f}")

    torch.save(model.state_dict(), f"./defensive_models/{archive_name}_final.pth")
    print(f"Final model for {archive_name} saved.")

if __name__ == "__main__":
    print("Loading BraTS data...")
    train_loader, val_loader, local_test_loader = get_brats_dataloaders(
        train_data_root='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count())
          # <- Optional if your loader supports this switch
    )
    print("BraTS DataLoaders ready.")

    # from DAE_model import DAE  # Your updated 2D DAE model

    print(f"\nCreating DAE Model I with input shape: {input_shape}")
    AE_I = DAE(
        image_shape=input_shape,
        structure=structure_AE_I,
        v_noise=0.1,
        activation=activation,
        reg_strength=reg_strength
    )
    train_autoencoder(AE_I, train_loader, val_loader, "BraTS_DAE2D_I")

    print(f"\nCreating DAE Model II with input shape: {input_shape}")
    AE_II = DAE(
        image_shape=input_shape,
        structure=structure_AE_II,
        v_noise=0.1,
        activation=activation,
        reg_strength=reg_strength
    )
    train_autoencoder(AE_II, train_loader, val_loader, "BraTS_DAE2D_II")

    print("\nTraining complete for both BraTS 2D Denoising Autoencoders!")
    print(f"Local Test Loader has {len(local_test_loader.dataset)} samples.")

