import torch
import numpy as np
import os
import pickle

class AttackDataBraTS:
    def __init__(self, data, labels, name="", directory="/kaggle/input/brats_data"):
        """
        Wrapper for BraTS volumetric data (normal or adversarial).
        
        data: Tensor [N, 4, 155, 240, 240] or path to saved object
        labels: Tensor or numpy array of segmentation masks [N, 155, 240, 240]
        name: Identifier string
        """
        if not isinstance(data, torch.Tensor):
            self.data = torch.tensor(np.array(data), dtype=torch.float32)
        else:
            self.data = data.float()
            
        # Ensure labels are tensor and long
        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        else:
            self.labels = labels.long()
            
        self.name = name
        self.length = len(self.data)
        self.directory = directory

    def print(self):
        return f"BraTS Attack:{self.name}"


def normalize_brats(images):
    """
    Normalize BraTS volumes to [0,1] range (assuming preprocessed data)
    images: Tensor [N, 4, 155, 240, 240]
    """
    return torch.clamp(images, 0.0, 1.0)


def prepare_brats_data(dataset, idx):
    images = []
    masks = []
    labels = []
    
    for i in idx:
        item = dataset[i]
        
        # Handle different return types
        if isinstance(item, tuple):
            img, mask = item[:2]  # Take first two elements
        elif isinstance(item, dict):
            img = item['image']
            mask = item['mask']
        else:
            raise ValueError(f"Unexpected dataset item type: {type(item)}")
        
        # Convert one-hot masks to class indices
        if mask.ndim == 4:  # 3D case [C, D, H, W]
            mask = mask.argmax(dim=0)  # Convert to [D, H, W]
        elif mask.ndim == 3:  # 2D case [C, H, W]
            mask = mask.argmax(dim=0)  # Convert to [H, W]
            
        images.append(img)
        masks.append(mask)
        labels.append(torch.any(mask > 0))  # Tumor presence

    X = torch.stack(images)  # [N, 4, 155, 240, 240] or [N, 4, 240, 240]
    targets = torch.stack(masks)  # [N, 155, 240, 240] or [N, 240, 240]
    Y = torch.tensor(labels, dtype=torch.long)
    
    return X, targets, Y

def save_obj(obj, name, directory='./brats_attack_data/'):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, directory='./brats_attack_data/'):
    if name.endswith(".pkl"):
        name = name[:-4]
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def generate_attack_data_brats(model, attack_type, epsilon, steps=10, num_samples=10, dataset=None):
    if dataset is None:
        raise ValueError("BraTS dataset required for attack generation")

    idx = torch.randperm(len(dataset))[:num_samples]
    X_clean, targets, Y = prepare_brats_data(dataset, idx)
    
    if attack_type.lower() == "fgsm":
        attacker = FGSM3DAttack(model, epsilon=epsilon)
    elif attack_type.lower() == "pgd":
        attacker = PGD3DAttack(model, eps=epsilon, alpha=epsilon/4, steps=steps)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    if X_clean.ndim == 5:  # 3D volumes [N, C, D, H, W]
        patch_size = (4, 64, 64, 64)
        X_adv = torch.empty_like(X_clean)
        target_patches = torch.empty((*X_clean.shape[:3], patch_size[2], patch_size[3]), dtype=torch.long)
        
        for i in range(0, X_clean.shape[2], patch_size[1]):
            for j in range(0, X_clean.shape[3], patch_size[2]):
                for k in range(0, X_clean.shape[4], patch_size[3]):
                    # Extract matching input and target patches
                    input_patch = X_clean[:, :, i:i+patch_size[1], j:j+patch_size[2], k:k+patch_size[3]]
                    target_patch = targets[:, i:i+patch_size[1], j:j+patch_size[2], k:k+patch_size[3]]
                    
                    adv_patch = attacker.generate(input_patch, target_patch)
                    X_adv[:, :, i:i+patch_size[1], j:j+patch_size[2], k:k+patch_size[3]] = adv_patch

    elif X_clean.ndim == 4:  # 2D slices [N, C, H, W]
        patch_size = (4, 64, 64)
        X_adv = torch.empty_like(X_clean)
        target_patches = torch.empty((X_clean.shape[0], patch_size[1], patch_size[2]), dtype=torch.long)
        
        for j in range(0, X_clean.shape[2], patch_size[1]):
            for k in range(0, X_clean.shape[3], patch_size[2]):
                # Extract matching input and target patches
                input_patch = X_clean[:, :, j:j+patch_size[1], k:k+patch_size[2]]
                target_patch = targets[:, j:j+patch_size[1], k:k+patch_size[2]]
                
                adv_patch = attacker.generate(input_patch, target_patch)
                X_adv[:, :, j:j+patch_size[1], k:k+patch_size[2]] = adv_patch

    else:
        raise ValueError(f"Unexpected input dimension: {X_clean.ndim}")
                
    return AttackDataBraTS(X_adv, targets, name=f"BraTS_{attack_type.upper()}_eps_{epsilon}")

class FGSM3DAttack:
    def __init__(self, model, epsilon=0.005):  # Reduced default epsilon for medical data
        self.model = model
        self.epsilon = epsilon
        self.device = next(model.parameters()).device
        
    def generate(self, volumes, masks):
        volumes = volumes.clone().detach().to(self.device)
        masks = masks.clone().detach().to(self.device)
        volumes.requires_grad = True
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = self.model(volumes)
            loss = F.cross_entropy(outputs, masks, reduction='none').mean()
        
        # Backward pass in full precision
        self.model.zero_grad()
        loss.backward()
        
        # Medical-adaptive perturbation scaling
        image_std = volumes.std().item()  # Get input statistics
        scaled_epsilon = self.epsilon * image_std
        
        # Constrained perturbation generation
        perturbation = scaled_epsilon * volumes.grad.data.sign()
        perturbation = torch.clamp(perturbation, -scaled_epsilon, scaled_epsilon)
        
        # Preserve anatomical validity
        perturbed_volumes = volumes + perturbation
        perturbed_volumes = torch.clamp(perturbed_volumes, 
                                      volumes.min().item(), 
                                      volumes.max().item()).detach()
        
        return perturbed_volumes

class PGD3DAttack:
    def __init__(self, model, eps=0.004, alpha=0.001, steps=10):  # Adjusted defaults
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.device = next(model.parameters()).device
        
    def generate(self, volumes, masks):
        """Medical-optimized PGD with tumor preservation checks"""
        orig_volumes = volumes.clone().detach().to(self.device)
        masks = masks.clone().detach().to(self.device)
        
        # Initialize with tumor-focused perturbation
        tumor_mask = (orig_volumes > 0.1).float()  # Simple tumor threshold
        delta = torch.randn_like(orig_volumes) * tumor_mask * self.eps
        adv_volumes = torch.clamp(orig_volumes + delta, 0, 1)
        
        for _ in range(self.steps):
            adv_volumes.requires_grad = True
            
            # Forward pass with medical constraints
            with torch.cuda.amp.autocast():
                outputs = self.model(adv_volumes)
                loss = F.cross_entropy(outputs, masks, reduction='none').mean()
            
            # Tumor-focused gradient calculation
            self.model.zero_grad()
            loss.backward()
            grad = adv_volumes.grad * tumor_mask  # Focus on tumor regions
            
            # Update with momentum for medical efficacy
            delta = delta + self.alpha * grad.sign()
            delta = torch.clamp(delta, -self.eps, self.eps)
            
            # Anatomical preservation
            adv_volumes = torch.clamp(orig_volumes + delta, 0, 1).detach()
            
        return adv_volumes

