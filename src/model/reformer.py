# reformer.py
# This file contains the SimpleReformer2D class, which is a simple model for image reconstruction.
# It is used to reconstruct the input image.
# The SimpleReformer2D class is a subclass of the nn.Module class, which is the base class for all neural network modules in PyTorch.
# The SimpleReformer2D class has the following methods:
# __init__: initializes the SimpleReformer2D model
# heal: reconstructs the input image

import torch

class SimpleReformer2D:
    def __init__(self, model_class, model_path, device='cuda', model_kwargs=None):
        self.device = torch.device(device)
        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def heal(self, X):
        """Medical-preserving reconstruction"""
        with torch.no_grad():
            X = X.to(self.device)
            recon = self.model(X)
            
            # Residual approach: only apply small corrections
            #residual = recon - X
            
            
            #alpha = 0.5  # Much smaller than before
            #healed = X + alpha * residual
            
            return torch.clamp(recon, 0, 1)
