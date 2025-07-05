# detector.py
# This file contains the AEDetector2D class, which is a deep autoencoder model for image denoising.
# It is used to detect anomalies in the input image.
# The AEDetector2D class is a subclass of the nn.Module class, which is the base class for all neural network modules in PyTorch.
# The AEDetector2D class has the following methods:
# __init__: initializes the AEDetector2D model
# mark: marks the input image with the anomaly score

import torch

class AEDetector2D:
    def __init__(self, model_class, model_path, p=2, device='cuda', model_kwargs=None):
        self.device = torch.device(device)
        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.p = p

    def mark(self, X):
        """X: 4D tensor [N, 4, 240, 240]"""
        with torch.no_grad():
            X = X.to(self.device)
            recon = self.model(X)
            error = torch.abs(X - recon)
            if self.p != 1:
                error = torch.pow(error, self.p)
            return torch.mean(error.view(X.size(0), -1), dim=1).cpu().numpy()
