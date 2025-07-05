import torch
import numpy as np
import albumentations as A
import binary
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

# --- 3. Albumentations Transforms ---
def get_train_transforms(size=240):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5
        ),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0)
        ], p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(std_range=(0.04, 0.2), mean_range=(0.0, 0.0), p=0.25),
    ])

def get_val_transforms(size=240):
    return A.Compose([
        A.Resize(size, size),
    ])

# --- 4. Segmentation Metrics ---
def dice_coefficient(y_true, y_pred):
    """
    Calculates the Dice coefficient.
    Args:
        y_true (np.array): Ground truth binary mask.
        y_pred (np.array): Predicted binary mask.
    Returns:
        float: Dice coefficient. Returns 1.0 if both masks are empty, 0.0 otherwise if one is empty.
    """
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 1.0
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return 0.0
    return binary.dc(y_pred.astype(bool), y_true.astype(bool))

def hausdorff_distance_95(y_true, y_pred):
    """
    Calculates the 95th percentile Hausdorff Distance.
    Args:
        y_true (np.array): Ground truth binary mask.
        y_pred (np.array): Predicted binary mask.
    Returns:
        float: 95th percentile Hausdorff Distance. Returns 0.0 if both masks are empty,
               np.inf if one is empty and the other is not.
    """
    if y_true.sum() == 0 and y_pred.sum() == 0:
        return 0.0
    if y_true.sum() == 0 or y_pred.sum() == 0:
        return np.inf
    return binary.hd95(y_pred.astype(bool), y_true.astype(bool))




class BraTSDataWrapper:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

def brats_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    return (images, masks)