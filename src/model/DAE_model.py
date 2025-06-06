import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, image_shape, structure, v_noise=0.0,
                 activation='relu', model_dir='./defensive_models/',
                 reg_strength=0.0):
        """
        Denoising autoencoder in PyTorch.

        image_shape: (C, H, W) format (e.g., (1, 28, 28))
        structure: List defining architecture. E.g., [32, 'max', 64, 'max']
        v_noise: Volume of Gaussian noise added during training
        activation: Activation function ('relu' or 'tanh')
        model_dir: Directory for saving/loading model
        reg_strength: L2 regularization strength (used in optimizer)
        """
        super(DenoisingAutoEncoder, self).__init__()
        self.image_shape = image_shape
        self.structure = structure
        self.v_noise = v_noise
        self.model_dir = model_dir
        self.activation_fn = getattr(F, activation)

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        layers = []
        in_channels = self.image_shape[0]

        for layer in self.structure:
            if isinstance(layer, int):
                layers.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
                in_channels = layer
            elif layer == 'max':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif layer == 'average':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                raise ValueError(f"Unknown layer type: {layer}")

        return nn.Sequential(*layers)

    def build_decoder(self):
        layers = []
        reversed_structure = list(reversed(self.structure))
        in_channels = self.get_last_conv_channels()

        for layer in reversed_structure:
            if isinstance(layer, int):
                layers.append(nn.Conv2d(in_channels, layer, kernel_size=3, padding=1))
                in_channels = layer
            elif layer == 'max' or layer == 'average':
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        # Final output layer
        out_channels = self.image_shape[0]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        return nn.Sequential(*layers)

    def get_last_conv_channels(self):
        # Get final number of filters from encoder
        channels = self.image_shape[0]
        for layer in self.structure:
            if isinstance(layer, int):
                channels = layer
        return channels

    def forward(self, x):
        x = self.encoder(x)
        x = torch.clamp(x, min=-5, max=5)
        x = self.activation_fn(x)
        x = self.decoder(x)
        return x

    def train_model(self, data, num_epochs=100, batch_size=256,
                    lr=0.001, weight_decay=0.0, if_save=True,
                    archive_name='dae.pth'):
        """
        data: an object with 'train_data' and 'validation_data' as Tensors (shape: [N, C, H, W])
        weight_decay: L2 regularization (passed to optimizer)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data.train_data, data.train_data),
            batch_size=batch_size, shuffle=True
        )

        val_data = data.validation_data.to(device)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for noisy_inputs, targets in train_loader:
                noisy_inputs = noisy_inputs.to(device)
                targets = targets.to(device)

                # Add noise
                noise = self.v_noise * torch.randn_like(noisy_inputs)
                noisy_inputs = torch.clamp(noisy_inputs + noise, 0., 1.)

                optimizer.zero_grad()
                outputs = self.forward(noisy_inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(val_data)
                val_loss = criterion(val_outputs, val_data).item()

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        if if_save:
            os.makedirs(self.model_dir, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(self.model_dir, archive_name))

    def load(self, archive_name, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir
        self.load_state_dict(torch.load(os.path.join(model_dir, archive_name)))
