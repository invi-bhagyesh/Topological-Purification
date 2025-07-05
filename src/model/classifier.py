# classifier.py
# This file contains the ClassifierUNet class, which is a UNet-based classifier for medical image segmentation/classification.
# It is used to classify the input image into different classes.
# The ClassifierUNet class is a subclass of the nn.Module class, which is the base class for all neural network modules in PyTorch.
# The ClassifierUNet class has the following methods:
# __init__: initializes the ClassifierUNet model
# classify: classifies the input image into different classes

import torch

class ClassifierUNet:
    def __init__(self, model_class, classifier_path, device='cuda', model_kwargs=None):
        """
        UNet-based classifier wrapper for medical image segmentation/classification.
        
        model_class: UNet architecture class
        classifier_path: Path to saved model weights (.pth)
        device: torch.device or string ("cuda"/"cpu")
        model_kwargs: Dictionary of kwargs for UNet initialization
        """
        self.path = classifier_path
        self.device = torch.device(device)
        
        # Initialize UNet with medical imaging defaults
        default_kwargs = {
            'in_channels': 4,        # For BraTS multi-modal input
            'out_channels': 4,       # Tumor sub-regions
            'features': [64, 128, 256]
        }
        combined_kwargs = {**default_kwargs, **(model_kwargs or {})}
        
        self.model = model_class(**combined_kwargs).to(self.device)
        self.model.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.model.eval()

    def classify(self, X, option="logit", T=1):
        """Returns flattened segmentation maps"""
        with torch.no_grad():
            # Move input to model's device
            X = X.to(self.device)
            
            if X.ndim == 5:  # 3D volumes [N, C, D, H, W]
                N, C, D, H, W = X.shape
                X = X.permute(0, 2, 1, 3, 4).reshape(N*D, C, H, W)
                
            outputs = self.model(X)  # Now both on same device
            
            if option == "logit":
                return outputs.view(-1).cpu().numpy()
            elif option == "prob":
                probs = F.softmax(outputs / T, dim=1)
                return probs.view(-1).cpu().numpy()

    def _aggregate_predictions(self, seg_map):
        """Convert segmentation map to class probabilities"""
        # Global average pooling over spatial dimensions
        return seg_map.mean(dim=[2, 3])  # [N, num_classes]

    def print(self):
        return f"ClassifierUNet++:{os.path.basename(self.path)}"
