import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from set_root_path import add_project_root
add_project_root()

from GAN import Encoder, Generator, Discriminator
from Topo_AE_BraTs import TopologicalSignatureDistance

def train_gan_topo_autoencoder(
    train_loader, val_loader, device,
    epochs=30, lr=1e-4, loss_type='mse',
    lambda_topo=1.0, topo_scale=0.01,
    start_delay_epochs=5
):
    latent_dim = 512
    encoder = Encoder(latent_dim=latent_dim).to(device)
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    recon_loss_fn = nn.MSELoss() if loss_type == 'mse' else nn.L1Loss()
    adv_loss_fn = nn.BCELoss()
    topo_loss_fn = TopologicalSignatureDistance().to(device)

    opt_EG = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=lr)
    opt_D = optim.Adam(discriminator.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(opt_EG, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        encoder.train()
        generator.train()
        discriminator.train()

        total_loss, total_recon, total_adv, total_topo = 0, 0, 0, 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            batch_size = images.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # --- Train Discriminator ---
            z = encoder(images)
            x_hat = generator(z).detach()
            D_real = discriminator(images)
            D_fake = discriminator(x_hat)

            d_loss = adv_loss_fn(D_real, real_labels) + adv_loss_fn(D_fake, fake_labels)
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # --- Train Encoder + Generator ---
            z = encoder(images)
            x_hat = generator(z)
            D_fake = discriminator(x_hat)

            recon_loss = recon_loss_fn(x_hat, images)
            adv_loss = adv_loss_fn(D_fake, real_labels)

            with torch.no_grad():
                x_dist = torch.cdist(images.view(batch_size, -1), images.view(batch_size, -1), p=2)
                z_dist = torch.cdist(z, z, p=2)
                x_dist = x_dist / x_dist.max()
                z_dist = z_dist / z_dist.max()

            topo_loss, _ = topo_loss_fn(x_dist, z_dist)
            topo_loss_scaled = topo_scale * topo_loss

            # Delay topo loss influence for initial epochs
            if epoch < start_delay_epochs:
                total_EG_loss = recon_loss + 1e-3 * adv_loss
            else:
                total_EG_loss = recon_loss + 1e-3 * adv_loss + lambda_topo * topo_loss_scaled

            opt_EG.zero_grad()
            total_EG_loss.backward()
            opt_EG.step()

            total_loss += total_EG_loss.item()
            total_recon += recon_loss.item()
            total_adv += adv_loss.item()
            total_topo += topo_loss_scaled.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}] | "
                    f"D_loss: {d_loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                    f"Adv: {adv_loss.item():.4f} | Topo: {topo_loss_scaled.item():.4f} | "
                    f"Total_G_loss: {total_EG_loss.item():.4f}"
                )

        # --- Validation (Topological only) ---
        encoder.eval()
        generator.eval()
        val_topo_loss = 0.0
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                z = encoder(val_images)
                recon = generator(z)
                batch_size = val_images.size(0)

                x_dist = torch.cdist(val_images.view(batch_size, -1), val_images.view(batch_size, -1), p=2)
                z_dist = torch.cdist(z, z, p=2)
                x_dist = x_dist / x_dist.max()
                z_dist = z_dist / z_dist.max()

                topo_loss_val, _ = topo_loss_fn(x_dist, z_dist)
                topo_loss_val_scaled = topo_scale * topo_loss_val
                val_topo_loss += topo_loss_val_scaled.item()

        val_topo_loss /= len(val_loader)
        scheduler.step(val_topo_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Topo Loss: {val_topo_loss:.4f}")

        # --- Save Best Model ---
        if val_topo_loss < best_val:
            best_val = val_topo_loss
            best_state = {
                'encoder': encoder.state_dict(),
                'generator': generator.state_dict()
            }
            print(f">>> Saved new best model (Val Topo = {best_val:.4f})")

    torch.save(best_state, 'gan_topo_recon_best.pth')
    print("✅ Best model saved to: gan_topo_recon_best.pth")
    return encoder, generator
