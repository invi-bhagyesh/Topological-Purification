import sys
# from data_setup import get_brats_dataloaders
# from DAE_model import DAE  # Assuming your DAE class is in DAE_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 50
reg_strength = 1e-9
activation = "relu"

# --- FIXED: Correct input shape for 2D slices ---
# Your dataset provides 2D slices with 4 channels: (C, H, W)
input_shape = (4, 240, 240)  # This is correct for 2D slice processing

# --- Architecture structures (for 2D operations) ---
structure_AE_I = [
    16,                  # Conv2d: 4 -> 16 channels
    "max",               # MaxPool2d: 240x240 -> 120x120
    32,                  # Conv2d: 16 -> 32 channels
    "max",               # MaxPool2d: 120x120 -> 60x60
    96,                  # Conv2d: 32 -> 96 channels
    "max",               # MaxPool2d: 60x60 -> 30x30
    "linear_bottleneck",
    512                  # Linear bottleneck size
]

structure_AE_II = [
    32,                  # Conv2d: 4 -> 32 channels
    "max",               # MaxPool2d: 240x240 -> 120x120
    64,                  # Conv2d: 32 -> 64 channels
    "max",               # MaxPool2d: 120x120 -> 60x60
    128,                 # Conv2d: 64 -> 128 channels
    "max",               # MaxPool2d: 60x60 -> 30x30
    256,                 # Conv2d: 128 -> 256 channels
    "max",               # MaxPool2d: 30x30 -> 15x15
    "linear_bottleneck",
    1024                 # Linear bottleneck size
]

def train_autoencoder(model, train_loader, val_loader, archive_name):
    """
    Trains a Denoising Autoencoder model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=reg_strength)
    criterion = nn.MSELoss()

    # Initialize wandb run
    wandb.init(
        project="brats_dae_training_2D",
        name=archive_name,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "regularization_strength": reg_strength,
            "activation": activation,
            "input_shape": input_shape,
            "structure_params": structure_AE_I if archive_name.endswith("I") else structure_AE_II
        }
    )

    best_val_loss = float('inf')

    print(f"\n--- Starting training for {archive_name} ---")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            imgs = batch_data['image']  # Shape: [B, 4, 240, 240]
            
            # Debug print for first batch
            if epoch == 0 and batch_idx == 0:
                print(f"Input batch shape: {imgs.shape}")  # Should be [B, 4, 240, 240]
                assert imgs.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {imgs.shape}"

            # Add Gaussian noise (using model's v_noise parameter)
            noisy_imgs = imgs + model.v_noise * torch.randn_like(imgs)
            noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)

            noisy_imgs, imgs = noisy_imgs.to(device), imgs.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, imgs)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                imgs = batch_data['image']
                
                # Add noise for validation consistency
                noisy_imgs = imgs + model.v_noise * torch.randn_like(imgs)
                noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)
                
                noisy_imgs, imgs = noisy_imgs.to(device), imgs.to(device)
                
                outputs = model(noisy_imgs)
                loss = criterion(outputs, imgs)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        # Logging
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "loss/train": avg_loss,
            "loss/val": avg_val_loss
        })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("./defensive_models/", exist_ok=True)
            torch.save(model.state_dict(), f"./defensive_models/{archive_name}_best.pth")
            print(f"  --> Best model for {archive_name} saved with Val Loss: {best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), f"./defensive_models/{archive_name}_final.pth")
    print(f"Final model for {archive_name} saved as ./defensive_models/{archive_name}_final.pth")

    wandb.finish()

if __name__ == "__main__":
    # Load BraTS data
    print("Loading BraTS data...")
    train_loader, val_loader, local_test_loader = get_brats_dataloaders(
        train_data_root='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count())  # Limit workers to avoid memory issues
    )
    print("BraTS DataLoaders ready.")

    # Train Model I
    print(f"\nCreating DAE Model I with input shape: {input_shape}")
    AE_I = DAE(
        image_shape=input_shape, 
        structure=structure_AE_I, 
        v_noise=0.1, 
        activation=activation, 
        reg_strength=reg_strength
    )
    train_autoencoder(AE_I, train_loader, val_loader, "BraTS_DAE_I")

    # Train Model II
    print(f"\nCreating DAE Model II with input shape: {input_shape}")
    AE_II = DAE(
        image_shape=input_shape, 
        structure=structure_AE_II, 
        v_noise=0.1, 
        activation=activation, 
        reg_strength=reg_strength
    )
    train_autoencoder(AE_II, train_loader, val_loader, "BraTS_DAE_II")

    print("\nTraining complete for both BraTS Denoising Autoencoders!")
    print(f"Local Test Loader has {len(local_test_loader.dataset)} samples for final evaluation.")