import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import random
from medpy.metric import binary

class MNIST:
    def __init__(self, batch_size=128):
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Automatically downloads if not available
        full_train = datasets.MNIST(root='/kaggle/working', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(root='/kaggle/working', train=False, download=True, transform=transform)

        # Split training and validation sets
        val_size = 5000
        train_size = len(full_train) - val_size
        self.train_data, self.validation_data = random_split(full_train, [train_size, val_size])

        # DataLoaders
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_data, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size)

    @staticmethod
    def print():
        return "MNIST dataset loaded."



class BraTSDataset(Dataset):
    def __init__(self, patients, transform=None, mode='train', slice_selection_method='all'):
        self.patients = patients
        self.transform = transform
        self.mode = mode
        self.slice_selection_method = slice_selection_method

        self.slices = []

        for patient in self.patients:
            patient_id = patient['id']
            patient_path = patient['path']
            grade = patient['grade']

            t1_path = os.path.join(patient_path, f"{patient_id}_t1.nii")
            t1ce_path = os.path.join(patient_path, f"{patient_id}_t1ce.nii")
            t2_path = os.path.join(patient_path, f"{patient_id}_t2.nii")
            flair_path = os.path.join(patient_path, f"{patient_id}_flair.nii")
            seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii")

            if not all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, seg_path]):
                t1_path = os.path.join(patient_path, f"{patient_id}_T1.nii.gz")
                t1ce_path = os.path.join(patient_path, f"{patient_id}_T1CE.nii.gz")
                t2_path = os.path.join(patient_path, f"{patient_id}_T2.nii.gz")
                flair_path = os.path.join(patient_path, f"{patient_id}_FLAIR.nii.gz")
                seg_path = os.path.join(patient_path, f"{patient_id}_seg.nii.gz")

            if not all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, seg_path]):
                print(f"Skipping {patient_id}: Missing required files")
                continue

            seg_img = nib.load(seg_path).get_fdata()

            if slice_selection_method == 'all':
                for slice_idx in range(seg_img.shape[2]):
                    if seg_img.shape[2] * 0.1 <= slice_idx <= seg_img.shape[2] * 0.9:
                        self.slices.append({
                            'patient': patient_id,
                            'grade': grade,
                            'slice_idx': slice_idx,
                            't1_path': t1_path,
                            't1ce_path': t1ce_path,
                            't2_path': t2_path,
                            'flair_path': flair_path,
                            'seg_path': seg_path
                        })
            elif slice_selection_method == 'tumor_only':
                for slice_idx in range(seg_img.shape[2]):
                    if np.any(seg_img[:, :, slice_idx] > 0):
                        self.slices.append({
                            'patient': patient_id,
                            'grade': grade,
                            'slice_idx': slice_idx,
                            't1_path': t1_path,
                            't1ce_path': t1ce_path,
                            't2_path': t2_path,
                            'flair_path': flair_path,
                            'seg_path': seg_path
                        })
            else:
                raise ValueError(f"Invalid slice selection method: {slice_selection_method}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_data = self.slices[idx]
        patient = slice_data['patient']
        slice_idx = slice_data['slice_idx']

        t1 = nib.load(slice_data['t1_path']).get_fdata()[:, :, slice_idx]
        t1ce = nib.load(slice_data['t1ce_path']).get_fdata()[:, :, slice_idx]
        t2 = nib.load(slice_data['t2_path']).get_fdata()[:, :, slice_idx]
        flair = nib.load(slice_data['flair_path']).get_fdata()[:, :, slice_idx]
        seg = nib.load(slice_data['seg_path']).get_fdata()[:, :, slice_idx]

        # Preprocess
        t1 = self._preprocess(t1)
        t1ce = self._preprocess(t1ce)
        t2 = self._preprocess(t2)
        flair = self._preprocess(flair)

        image = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)

        mask = np.zeros((3, *seg.shape), dtype=np.float32)
        mask[0, seg == 1] = 1  # NCR/NET
        mask[1, seg == 2] = 1  # ED
        mask[2, seg == 4] = 1  # ET

        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0), mask=mask.transpose(1, 2, 0))
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask'].transpose(2, 0, 1)

        return {
            'image': torch.from_numpy(image),
            'mask': torch.from_numpy(mask),
            'patient': patient,
            'slice': slice_idx
        }

    def _preprocess(self, img):
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std

        img = np.clip(img, -5, 5)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        return img

def get_train_transforms(size=240):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
    ])

def get_val_transforms(size=240):
    return A.Compose([
        A.Resize(size, size),
    ])

def dice_coefficient(y_true, y_pred):
    return binary.dc(y_pred, y_true)

def hausdorff_distance_95(y_true, y_pred):
    return binary.hd95(y_pred, y_true)
