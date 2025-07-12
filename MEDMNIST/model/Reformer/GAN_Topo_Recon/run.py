import argparse
import torch
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from set_root_path import add_project_root
add_project_root()  # This adds topo/ to sys.path

from MEDMNIST.dataloader import load_medmnist

from GAN import Encoder, Generator
from train import train_gan_topo_autoencoder

def get_gan_data_loaders(batch_size, data_flag):
    train_loader, val_loader, _, info = load_medmnist(data_flag=data_flag, batch_size=batch_size)
    return train_loader, val_loader, info

def main():
    parser = argparse.ArgumentParser(description='Train or load GAN-based autoencoder for adversarial purification')
    parser.add_argument('--train_gan', action='store_true', help='Train the GAN autoencoder')
    parser.add_argument('--epochs_gan', type=int, default=30, help='Epochs for GAN training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr_gan', type=float, default=1e-4, help='Learning rate for GAN')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='Reconstruction loss')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--data_flag', type=str, default='pathmnist', help='MedMNIST dataset flag')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, info = get_gan_data_loaders(args.batch_size, args.data_flag)

    if args.train_gan:
        print("Training GAN-based reconstructor...")
        encoder, generator = train_gan_topo_autoencoder(
            train_loader, val_loader, device,
            epochs=args.epochs_gan,
            lr=args.lr_gan,
            loss_type=args.loss_type,
            lambda_topo=1.0  # You can expose as CLI arg
        )

    else:
        print("Loading pretrained reconstructor from gan_recon.pth")
        n_channels = info.get('n_channels', 3)  # Default to 3 channels if not specified
        encoder = Encoder(input_channels=n_channels).to(device)
        generator = Generator(output_channels=n_channels).to(device)
        checkpoint = torch.load('gan_recon.pth', map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        generator.load_state_dict(checkpoint['generator'])
        encoder.eval()
        generator.eval()
        print("Loaded GAN reconstructor.")

    # Optional: Save for downstream use
    torch.save({
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict()
    }, 'gan_recon.pth')

if __name__ == '__main__':
    main()
