# DAE.py
# Enhanced Deep Autoencoder with task-specific functionality
# Supports: segmentation, multiclass classification, binary classification
# Loss options: reconstruction, SimCLR contrastive loss
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DAE(nn.Module):
    def __init__(self, image_shape, structure, task_type='segmentation', dataset='brats', 
                 num_classes=None, v_noise=0.1, activation=nn.ReLU, reg_strength=1e-4,
                 loss_type='reconstruction', projection_dim=128, temperature=0.1):
        """
        Enhanced DAE supporting multiple tasks and loss types.
        
        Args:
            image_shape: Input image shape (C, H, W)
            structure: Network structure definition
            task_type: 'segmentation', 'multiclass', 'binary'
            dataset: 'brats', 'cifar10', 'imagenet', etc.
            num_classes: Number of classes for classification tasks
            v_noise: Noise variance for denoising
            activation: Activation function
            reg_strength: Regularization strength
            loss_type: 'reconstruction' or 'simclr'
            projection_dim: Dimension for SimCLR projection head
            temperature: Temperature for SimCLR loss
        """
        super(DAE, self).__init__()
        
        self.image_shape = image_shape
        self.structure = structure
        self.task_type = task_type.lower()
        self.dataset = dataset.lower()
        self.num_classes = num_classes
        self.v_noise = v_noise
        self.reg_strength = reg_strength
        self.loss_type = loss_type.lower()
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Validate task configuration
        self._validate_config()
        
        # Handle activation parameter
        if isinstance(activation, str):
            activation_map = {
                'relu': nn.ReLU,
                'leaky_relu': nn.LeakyReLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'elu': nn.ELU,
                'gelu': nn.GELU
            }
            activation_fn = activation_map.get(activation.lower(), nn.ReLU)
        else:
            activation_fn = activation
        
        # Extract channel information and special operations
        channels = []
        operations = []
        
        for item in structure:
            if isinstance(item, int):
                channels.append(item)
                operations.append('conv')
            elif item == "max":
                operations.append('maxpool')
            elif item == "linear_bottleneck":
                operations.append('linear_bottleneck')
        
        # Build encoder
        self.encoder_layers = nn.ModuleList()
        
        in_channels = image_shape[0]
        current_size = image_shape[1]
        
        channel_idx = 0
        for i, op in enumerate(operations):
            if op == 'conv':
                out_channels = channels[channel_idx]
                self.encoder_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                self.encoder_layers.append(activation_fn(inplace=True))
                in_channels = out_channels
                channel_idx += 1
                
            elif op == 'maxpool':
                self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_size = current_size // 2
                
            elif op == 'linear_bottleneck':
                flattened_size = in_channels * current_size * current_size
                bottleneck_size = channels[channel_idx]
                
                self.encoder_layers.append(nn.Flatten())
                self.encoder_layers.append(nn.Linear(flattened_size, bottleneck_size))
                self.encoder_layers.append(activation_fn(inplace=True))
                
                self.bottleneck_size = bottleneck_size
                self.pre_flatten_channels = in_channels
                self.pre_flatten_size = current_size
                channel_idx += 1
                break
        
        # Build task-specific heads
        self._build_task_heads(activation_fn)
        
        # Build decoder (only for reconstruction tasks)
        if self.loss_type == 'reconstruction':
            self._build_decoder(channels, operations, activation_fn)
        
        # Build SimCLR projection head
        if self.loss_type == 'simclr':
            self._build_projection_head(activation_fn)
    
    def _validate_config(self):
        """Validate task and dataset configuration."""
        if self.task_type in ['multiclass', 'binary'] and self.num_classes is None:
            raise ValueError(f"num_classes must be specified for {self.task_type} classification")
        
        if self.task_type == 'binary' and self.num_classes != 2:
            raise ValueError("Binary classification requires num_classes=2")
        
        # Dataset-specific validations
        if self.dataset == 'brats' and self.task_type != 'segmentation':
            print("Warning: BraTS dataset is typically used for segmentation tasks")
        
        if self.dataset == 'cifar10' and self.task_type == 'segmentation':
            print("Warning: CIFAR-10 is typically used for classification tasks")
    
    def _build_task_heads(self, activation_fn):
        """Build task-specific output heads."""
        if self.task_type == 'segmentation':
            # For segmentation, we need pixel-wise classification
            if hasattr(self, 'bottleneck_size'):
                # If we have a bottleneck, we'll need to upsample in decoder
                self.segmentation_head = None  # Will be built in decoder
            else:
                # Direct segmentation head
                last_channels = self.structure[-1] if isinstance(self.structure[-1], int) else self.structure[-2]
                self.segmentation_head = nn.Conv2d(last_channels, self.num_classes or 4, kernel_size=1)
        
        elif self.task_type in ['multiclass', 'binary']:
            # For classification, we need a global classifier
            if hasattr(self, 'bottleneck_size'):
                # Use bottleneck features
                feature_dim = self.bottleneck_size
            else:
                # Use global average pooling
                last_channels = self.structure[-1] if isinstance(self.structure[-1], int) else self.structure[-2]
                feature_dim = last_channels
                self.global_pool = nn.AdaptiveAvgPool2d(1)
            
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                activation_fn(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(feature_dim // 2, self.num_classes)
            )
    
    def _build_decoder(self, channels, operations, activation_fn):
        """Build decoder for reconstruction tasks."""
        self.decoder_layers = nn.ModuleList()
        
        if 'linear_bottleneck' in operations:
            reconstruct_size = self.pre_flatten_channels * self.pre_flatten_size * self.pre_flatten_size
            self.decoder_layers.append(nn.Linear(self.bottleneck_size, reconstruct_size))
            self.decoder_layers.append(activation_fn(inplace=True))
            
            current_channels = self.pre_flatten_channels
            current_size = self.pre_flatten_size
        else:
            current_channels = channels[-1]
        
        reversed_channels = channels[:-1] if 'linear_bottleneck' in operations else channels[:-1]
        reversed_channels.reverse()
        
        # Determine output channels based on task
        if self.task_type == 'segmentation':
            output_channels = self.num_classes or self.image_shape[0]
        else:
            output_channels = self.image_shape[0]
        
        reversed_channels.append(output_channels)
        
        num_maxpools = operations.count('maxpool')
        
        for i in range(len(reversed_channels)):
            if i < num_maxpools:
                self.decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
            out_channels = reversed_channels[i]
            self.decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=3, padding=1))
            
            if i < len(reversed_channels) - 1:
                self.decoder_layers.append(activation_fn(inplace=True))
            elif self.task_type == 'segmentation':
                # Add softmax for segmentation
                self.decoder_layers.append(nn.Softmax(dim=1))
            
            current_channels = out_channels
    
    def _build_projection_head(self, activation_fn):
        """Build projection head for SimCLR."""
        if hasattr(self, 'bottleneck_size'):
            feature_dim = self.bottleneck_size
        else:
            last_channels = self.structure[-1] if isinstance(self.structure[-1], int) else self.structure[-2]
            feature_dim = last_channels
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            activation_fn(inplace=True),
            nn.Linear(feature_dim, self.projection_dim)
        )
    
    def add_noise(self, x):
        """Add noise for denoising."""
        if self.training and self.v_noise > 0:
            noise = torch.randn_like(x) * self.v_noise
            return x + noise
        return x
    
    def encode(self, x):
        """Encode input to latent representation."""
        x = self.add_noise(x)
        
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode(self, x):
        """Decode latent representation."""
        if not hasattr(self, 'decoder_layers'):
            raise ValueError("Decoder not available for SimCLR loss type")
        
        need_reshape = False
        reshape_channels = None
        reshape_size = None
        
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, nn.Linear) and i == 0:
                need_reshape = True
                reshape_channels = self.pre_flatten_channels
                reshape_size = self.pre_flatten_size
                
            x = layer(x)
            
            if need_reshape and isinstance(layer, nn.Linear):
                x = x.view(-1, reshape_channels, reshape_size, reshape_size)
                need_reshape = False
        
        return x
    
    def forward(self, x):
        """Forward pass."""
        # Encode
        encoded = self.encode(x)
        
        if self.loss_type == 'reconstruction':
            # Reconstruction path
            if self.task_type == 'segmentation':
                return self.decode(encoded)
            elif self.task_type in ['multiclass', 'binary']:
                # For classification with reconstruction, return both
                reconstructed = self.decode(encoded)
                
                # Get features for classification
                if hasattr(self, 'global_pool'):
                    # Use encoded features before bottleneck
                    features = encoded
                    if len(features.shape) == 4:  # If still spatial
                        features = self.global_pool(features).flatten(1)
                else:
                    features = encoded
                
                classification = self.classifier(features)
                return {'reconstruction': reconstructed, 'classification': classification}
            else:
                return self.decode(encoded)
        
        elif self.loss_type == 'simclr':
            # SimCLR path
            if hasattr(self, 'global_pool') and len(encoded.shape) == 4:
                encoded = self.global_pool(encoded).flatten(1)
            
            projections = self.projection_head(encoded)
            return F.normalize(projections, dim=1)
    
    def get_loss(self, outputs, targets=None, **kwargs):
        """Calculate task and loss-type specific loss."""
        if self.loss_type == 'reconstruction':
            return self._get_reconstruction_loss(outputs, targets)
        elif self.loss_type == 'simclr':
            return self._get_simclr_loss(outputs, **kwargs)
    
    def _get_reconstruction_loss(self, outputs, targets):
        """Calculate reconstruction loss."""
        if self.task_type == 'segmentation':
            return F.cross_entropy(outputs, targets)
        elif self.task_type in ['multiclass', 'binary']:
            recon_loss = F.mse_loss(outputs['reconstruction'], targets['images'])
            if self.task_type == 'binary':
                class_loss = F.binary_cross_entropy_with_logits(
                    outputs['classification'], targets['labels'].float()
                )
            else:
                class_loss = F.cross_entropy(outputs['classification'], targets['labels'])
            return recon_loss + class_loss
        else:
            return F.mse_loss(outputs, targets)
    
    def _get_simclr_loss(self, projections, **kwargs):
        """Calculate SimCLR contrastive loss."""
        batch_size = projections.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(projections, projections.T) / self.temperature
        
        # Create positive pairs mask (assuming augmented pairs are adjacent)
        pos_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=projections.device)
        for i in range(0, batch_size, 2):
            if i + 1 < batch_size:
                pos_mask[i, i + 1] = True
                pos_mask[i + 1, i] = True
        
        # Create negative mask (all pairs except self and positive)
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=projections.device) & ~pos_mask
        
        # Calculate loss
        pos_sim = sim_matrix[pos_mask]
        neg_sim = sim_matrix[neg_mask].view(batch_size, -1)
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=projections.device)
        
        return F.cross_entropy(logits, labels)
