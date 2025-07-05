import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
import argparse
from model.DAE import DAE
from data.dataloader import get_brats_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Train DAE models for BraTS data')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, 
                       default='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
                       help='Path to BraTS data root directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=8, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate for optimizer')
    parser.add_argument('--reg_strength', type=float, default=1e-9, 
                       help='Weight decay/regularization strength')
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['relu', 'leaky_relu', 'tanh'], 
                       help='Activation function')
    
    # Model parameters
    parser.add_argument('--input_shape', type=str, default='4,240,240', 
                       help='Input shape as comma-separated values (C,H,W)')
    parser.add_argument('--v_noise', type=float, default=0.1, 
                       help='Noise level for denoising autoencoder')
    
    # Model architecture parameters
    parser.add_argument('--model_type', type=str, default='both', 
                       choices=['I', 'II', 'both'], 
                       help='Which DAE model to train (I, II, or both)')
    
    # Structure parameters
    parser.add_argument('--structure_I', type=str, 
                       default='32,max,64,max,128,max,linear_bottleneck,2048',
                       help='Structure for Model I as comma-separated values (numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck)')
    parser.add_argument('--structure_II', type=str, 
                       default='64,max,128,max,256,max,512',
                       help='Structure for Model II as comma-separated values (numbers for channels, "max" for maxpool)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./defensive_models/', 
                       help='Directory to save trained models')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Enable wandb logging')
    
    return parser.parse_args()

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_structure(structure_str):
    """Parse structure string into list of integers and strings"""
    structure = []
    for item in structure_str.split(','):
        item = item.strip()
        if item == 'max':
            structure.append('max')
        elif item == 'linear_bottleneck':
            structure.append('linear_bottleneck')
        else:
            try:
                structure.append(int(item))
            except ValueError:
                raise ValueError(f"Invalid structure item: {item}. Must be integer, 'max', or 'linear_bottleneck'")
    return structure

# --- Architecture structures for 2D Conv DAE ---
# Default structures are defined in parse_args() function

def train_autoencoder(model, train_loader, val_loader, archive_name, args):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.reg_strength)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print(f"\n--- Starting training for {archive_name} ---")
    print(f"Device: {device}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    for epoch in range(args.epochs):
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
                print(f"Epoch [{epoch + 1}/{args.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

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

        print(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/{archive_name}_best.pth")
            print(f"  --> Best model for {archive_name} saved with Val Loss: {best_val_loss:.4f}")

    torch.save(model.state_dict(), f"{args.output_dir}/{archive_name}_final.pth")
    print(f"Final model for {archive_name} saved.")

def main():
    args = parse_args()
    
    # Parse input shape and structures
    input_shape = tuple(map(int, args.input_shape.split(',')))
    structure_I = parse_structure(args.structure_I)
    structure_II = parse_structure(args.structure_II)
    
    print("Loading BraTS data...")
    train_loader, val_loader, local_test_loader = get_brats_dataloaders(
        train_data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=min(args.num_workers, os.cpu_count())
    )
    print("BraTS DataLoaders ready.")

    print(f"\nCreating DAE models with input shape: {input_shape}")
    print(f"Model I structure: {structure_I}")
    print(f"Model II structure: {structure_II}")
    
    if args.model_type in ['I', 'both']:
        print(f"\nCreating DAE Model I with input shape: {input_shape}")
        AE_I = DAE(
            image_shape=input_shape,
            structure=structure_I,
            v_noise=args.v_noise,
            activation=args.activation,
            reg_strength=args.reg_strength
        )
        train_autoencoder(AE_I, train_loader, val_loader, "BraTS_DAE2D_I", args)

    if args.model_type in ['II', 'both']:
        print(f"\nCreating DAE Model II with input shape: {input_shape}")
        AE_II = DAE(
            image_shape=input_shape,
            structure=structure_II,
            v_noise=args.v_noise,
            activation=args.activation,
            reg_strength=args.reg_strength
        )
        train_autoencoder(AE_II, train_loader, val_loader, "BraTS_DAE2D_II", args)

    print("\nTraining complete!")
    print(f"Local Test Loader has {len(local_test_loader.dataset)} samples.")

if __name__ == "__main__":
    main()

