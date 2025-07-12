import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from set_root_path import add_project_root
add_project_root()

from MEDMNIST.dataloader import load_medmnist
from GAN import Encoder, Generator, Discriminator

def train_gan_autoencoder(train_loader, val_loader, device, epochs=30, lr=1e-4, loss_type='mse'):
    latent_dim = 512
    encoder = Encoder(latent_dim=latent_dim).to(device)
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    recon_loss_fn = nn.MSELoss() if loss_type == 'mse' else nn.L1Loss()
    adv_loss_fn = nn.BCELoss()

    opt_EG = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=lr)
    opt_D = optim.Adam(discriminator.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt_EG, mode='min', factor=0.5, patience=10)

    for epoch in range(epochs):
        encoder.train()
        generator.train()
        discriminator.train()

        total_loss, total_recon, total_adv, total_d = 0, 0, 0, 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.size(0)

            # Use label smoothing
            real_labels = torch.full((batch_size, 1), 0.9, device=device)
            fake_labels = torch.full((batch_size, 1), 0.1, device=device)

            # -------------------
            # Train Discriminator (every 2 batches)
            # -------------------
            if batch_idx % 2 == 0:
                z = encoder(images)
                x_hat = generator(z).detach()
                D_real = discriminator(images)
                D_fake = discriminator(x_hat)

                d_loss = adv_loss_fn(D_real, real_labels) + adv_loss_fn(D_fake, fake_labels)
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()
            else:
                d_loss = torch.tensor(0.0)

            # -------------------
            # Train Encoder + Generator
            # -------------------
            z = encoder(images)
            x_hat = generator(z)
            D_fake = discriminator(x_hat)

            recon_loss = recon_loss_fn(x_hat, images)
            adv_loss = adv_loss_fn(D_fake, real_labels)

            total_EG_loss = recon_loss + 1e-3 * adv_loss
            opt_EG.zero_grad()
            total_EG_loss.backward()
            opt_EG.step()

            total_loss += total_EG_loss.item()
            total_recon += recon_loss.item()
            total_adv += adv_loss.item()
            total_d += d_loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}] | "
                    f"D_loss: {d_loss.item():.4f} | "
                    f"G_recon: {recon_loss.item():.4f} | G_adv: {adv_loss.item():.4f} | "
                    f"Total_G_loss: {total_EG_loss.item():.4f}"
                )

        # Validation (reconstruction loss only)
        encoder.eval()
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                z = encoder(val_images)
                recon = generator(z)
                val_loss += recon_loss_fn(recon, val_images).item()

        val_loss /= len(val_loader)
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {total_loss/len(train_loader):.4f}, "
            f"Val Recon Loss: {val_loss:.4f}, "
            f"Avg D_loss: {total_d/len(train_loader):.4f}"
        )
        scheduler.step(val_loss)

    torch.save({
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict()
    }, 'gan_recon.pth')
    print("✅ GAN autoencoder saved to gan_recon.pth")
    return encoder, generator

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _, _ = load_medmnist(batch_size=32)
    train_gan_autoencoder(train_loader, val_loader, device)
