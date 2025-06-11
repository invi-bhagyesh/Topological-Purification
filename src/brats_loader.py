# data_setup.py

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm 
from medpy.metric import binary # For Dice and Hausdorff metrics


# --- 1. Helper Function to Collect Patient Information ---
def collect_patient_info_from_root(data_root_path, grade_subfolders=True):
    """
    Scans a given BraTS root directory to collect patient information.

    Args:
        data_root_path (str): The root path to the BraTS data.
        grade_subfolders (bool): If True, expects 'HGG' and 'LGG' subfolders.
                                 If False, expects patient folders directly under data_root_path.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'id', 'path', and 'grade' for a patient.
    """
    patients_info = []

    if grade_subfolders:
        hgg_path = os.path.join(data_root_path, 'HGG')
        lgg_path = os.path.join(data_root_path, 'LGG')

        # Process HGG patients
        if os.path.exists(hgg_path):
            for patient_folder_name in os.listdir(hgg_path):
                patient_full_path = os.path.join(hgg_path, patient_folder_name)
                if os.path.isdir(patient_full_path):
                    patients_info.append({
                        'id': patient_folder_name,
                        'path': patient_full_path,
                        'grade': 'HGG'
                    })
        # else: # Optional: Uncomment for warnings if paths are missing
        #     print(f"Warning: HGG path not found in {data_root_path}: {hgg_path}")

        # Process LGG patients
        if os.path.exists(lgg_path):
            for patient_folder_name in os.listdir(lgg_path):
                patient_full_path = os.path.join(lgg_path, patient_folder_name)
                if os.path.isdir(patient_full_path):
                    patients_info.append({
                        'id': patient_folder_name,
                        'path': patient_full_path,
                        'grade': 'LGG'
                    })
        # else: # Optional: Uncomment for warnings if paths are missing
        #     print(f"Warning: LGG path not found in {data_root_path}: {lgg_path}")
    else: # No HGG/LGG subfolders, patient folders are direct children
        # print(f"Attempting to find patient folders directly under: {data_root_path}") # Optional print
        if os.path.exists(data_root_path):
            for patient_folder_name in os.listdir(data_root_path):
                patient_full_path = os.path.join(data_root_path, patient_folder_name)
                # Ensure it's a directory and looks like a patient folder (e.g., starts with BraTS)
                if os.path.isdir(patient_full_path) and patient_folder_name.startswith('BraTS'):
                    patients_info.append({
                        'id': patient_folder_name,
                        'path': patient_full_path,
                        'grade': 'Unknown' # Placeholder grade for validation/test set
                    })
        # else: # Optional: Uncomment for warnings if paths are missing
        #     print(f"Error: Data root path not found: {data_root_path}")

    return patients_info


# --- 2. BraTSDataset Class (YOUR ORIGINAL VERSION) ---
# This version implies that segmentation masks MUST be present for all patients.
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

            seg_img = nib.load(seg_path).get_fdata() # This line implicitly expects seg_path to exist

            if slice_selection_method == 'all':
                for slice_idx in range(seg_img.shape[2]):
                    # Include slices within the central 80% range
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
            # Albumentations expects (H, W, C) for image and mask
            transformed = self.transform(image=image.transpose(1, 2, 0), mask=mask.transpose(1, 2, 0))
            image = transformed['image'].transpose(2, 0, 1) # Back to (C, H, W)
            mask = transformed['mask'].transpose(2, 0, 1)   # Back to (C, H, W)

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


# --- Main function to get DataLoaders ---
def get_brats_dataloaders(
    train_data_root='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
    train_ratio=0.70,
    val_ratio=0.15,
    local_test_ratio=0.15,
    random_state=42,
    batch_size=4,
    num_workers=os.cpu_count()
):
    """
    Collects patient information and divides it into training, validation,
    and a local test set, returning corresponding PyTorch DataLoaders.

    Args:
        train_data_root (str): Path to the BraTS 2019 training data.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        local_test_ratio (float): Proportion of data for the local test set.
        random_state (int): Seed for reproducible splits.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, local_test_loader)
    """
    if not np.isclose(train_ratio + val_ratio + local_test_ratio, 1.0):
        raise ValueError("Train, Val, and Local Test ratios must sum to 1.0")

    print(f"Collecting patient information from: {train_data_root}")
    all_patients = collect_patient_info_from_root(train_data_root, grade_subfolders=True)
    print(f"Collected info for {len(all_patients)} total patients.")

    # Step 1: Separate out the local test set first
    train_val_patients, local_test_patients = train_test_split(
        all_patients,
        test_size=local_test_ratio,
        random_state=random_state
    )
    print(f"Split: {len(train_val_patients)} patients for train/val, {len(local_test_patients)} patients for local test.")

    # Step 2: Divide the remaining into train and validation
    # Adjust validation ratio based on the new size of `train_val_patients`
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)

    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=adjusted_val_ratio,
        random_state=random_state
    )
    print(f"Further split: {len(train_patients)} patients for training, {len(val_patients)} patients for validation.")

    # Initialize Datasets
    train_dataset = BraTSDataset(
        patients=train_patients,
        transform=get_train_transforms(),
        mode='train',
        slice_selection_method='tumor_only'
    )
    val_dataset = BraTSDataset(
        patients=val_patients,
        transform=get_val_transforms(),
        mode='val',
        slice_selection_method='all'
    )
    local_test_dataset = BraTSDataset(
        patients=local_test_patients,
        transform=get_val_transforms(),
        mode='test',
        slice_selection_method='all'
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    local_test_loader = DataLoader(local_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\n--- DataLoaders Summary ---")
    print(f"Train Loader: {len(train_dataset)} slices, Batch Size: {train_loader.batch_size}")
    print(f"Validation Loader: {len(val_dataset)} slices, Batch Size: {val_loader.batch_size}")
    print(f"Local Test Loader: {len(local_test_dataset)} slices, Batch Size: {local_test_loader.batch_size}")

    return train_loader, val_loader, local_test_loader

#  if you run data_setup.py directly 
if __name__ == '__main__':
    train_loader, val_loader, local_test_loader = get_brats_dataloaders(
        train_data_root='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
        batch_size=8 # Example batch size
    )
    print("\nDataLoaders created successfully for local testing!")
    # You can further inspect loaders here, e.g., next(iter(train_loader))