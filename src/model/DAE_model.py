class DAE(nn.Module):
    def __init__(self, image_shape, structure, v_noise=0.1, activation=nn.ReLU, reg_strength=1e-4):
        super(DAE, self).__init__()
        
        self.image_shape = image_shape
        self.structure = structure
        self.v_noise = v_noise
        self.reg_strength = reg_strength
        
        # Handle activation parameter - convert string to actual activation class if needed
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
        
        in_channels = image_shape[0]  # Initial input channels (4 for BraTS)
        current_size = image_shape[1]  # Assuming square images (240x240)
        
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
                # Flatten and create bottleneck
                flattened_size = in_channels * current_size * current_size
                bottleneck_size = channels[channel_idx]
                
                self.encoder_layers.append(nn.Flatten())
                self.encoder_layers.append(nn.Linear(flattened_size, bottleneck_size))
                self.encoder_layers.append(activation_fn(inplace=True))
                
                # Store info for decoder
                self.bottleneck_size = bottleneck_size
                self.pre_flatten_channels = in_channels
                self.pre_flatten_size = current_size
                channel_idx += 1
                break
        
        # Build decoder (reverse of encoder)
        self.decoder_layers = nn.ModuleList()
        
        # Handle bottleneck reconstruction
        if 'linear_bottleneck' in [op for op in operations]:
            # Reverse the linear bottleneck
            reconstruct_size = self.pre_flatten_channels * self.pre_flatten_size * self.pre_flatten_size
            self.decoder_layers.append(nn.Linear(self.bottleneck_size, reconstruct_size))
            self.decoder_layers.append(activation_fn(inplace=True))
            
            # Reshape back to feature maps
            # This will be handled in forward pass
            current_channels = self.pre_flatten_channels
            current_size = self.pre_flatten_size
        else:
            current_channels = channels[-1]
        
        # Reverse the convolutional layers
        reversed_channels = channels[:-1] if 'linear_bottleneck' in operations else channels[:-1]
        reversed_channels.reverse()
        reversed_channels.append(image_shape[0])  # Back to original input channels
        
        # Count maxpool operations to know how many upsample layers we need
        num_maxpools = operations.count('maxpool')
        
        # Add transposed convolutions and upsampling
        for i in range(len(reversed_channels)):
            if i < num_maxpools:
                # Add upsampling first
                self.decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
            out_channels = reversed_channels[i]
            self.decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=3, padding=1))
            
            # Don't add activation after the final layer
            if i < len(reversed_channels) - 1:
                self.decoder_layers.append(activation_fn(inplace=True))
            
            current_channels = out_channels
    
    def add_noise(self, x):
        if self.training and self.v_noise > 0:
            noise = torch.randn_like(x) * self.v_noise
            return x + noise
        return x
    
    def forward(self, x):
        # Add noise for denoising
        x = self.add_noise(x)
        
        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Decoder
        need_reshape = False
        reshape_channels = None
        reshape_size = None
        
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, nn.Linear) and i == 0:
                # Store info for reshaping after linear layer
                need_reshape = True
                reshape_channels = self.pre_flatten_channels
                reshape_size = self.pre_flatten_size
                
            x = layer(x)
            
            # Reshape after first linear layer in decoder
            if need_reshape and isinstance(layer, nn.Linear):
                x = x.view(-1, reshape_channels, reshape_size, reshape_size)
                need_reshape = False
        
        return x