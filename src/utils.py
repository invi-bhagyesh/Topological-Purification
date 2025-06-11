import pickle
import os
import torch
import numpy as np # Added for potential numpy array loading if your data is in .npy format

def prepare_data(dataset, idx):
    """
    Extracts image and label data at specific indices from the test set.
    This function is specifically designed for MNIST-like datasets.

    dataset: Instance of MNIST() class.
    idx: List or tensor of indices into the test set.
    Returns:
        X: torch.Tensor of images [N, 1, 28, 28]
        targets: torch.Tensor of integer labels [N]
        Y: Same as targets (for compatibility with original code using argmax)
    """
    # dataset.test_data is a torchvision.datasets.MNIST object
    images = []
    labels = []
    for i in idx:
        img, label = dataset.test_data[i]
        images.append(img)
        labels.append(label)

    X = torch.stack(images)  # shape: [N, 1, 28, 28]
    targets = torch.tensor(labels, dtype=torch.long)  # shape: [N]
    Y = targets.clone()  # No one-hot, so argmax would just return same

    return X, targets, Y


def save_obj(obj, name, directory='./attack_data/'):
    """
    Saves an object using pickle.
    For large PyTorch tensors, consider using `torch.save(tensor, 'path/to/file.pt')` directly
    in your code where the tensor is created/saved, as it's often more efficient.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, name + '.pkl')
    print(f"Saving object to: {file_path}")
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, directory='./attack_data/'):
    """
    Loads an object from a specified directory.
    It attempts to load from .pkl, .pt/.pth (PyTorch), and .npy (NumPy) formats.
    """
    # Construct full paths for different possible extensions
    file_path_pkl = os.path.join(directory, name + '.pkl')
    file_path_pt = os.path.join(directory, name + '.pt')
    file_path_pth = os.path.join(directory, name + '.pth') # Often used interchangeably with .pt
    file_path_npy = os.path.join(directory, name + '.npy') # For NumPy arrays

    # Check for existing files in order of preference
    if os.path.exists(file_path_pt):
        print(f"Loading PyTorch tensor from: {file_path_pt}")
        return torch.load(file_path_pt)
    elif os.path.exists(file_path_pth):
        print(f"Loading PyTorch tensor from: {file_path_pth}")
        return torch.load(file_path_pth)
    elif os.path.exists(file_path_npy):
        print(f"Loading NumPy array from: {file_path_npy}")
        return np.load(file_path_npy)
    elif os.path.exists(file_path_pkl):
        print(f"Loading pickle object from: {file_path_pkl}")
        with open(file_path_pkl, 'rb') as f:
            return pickle.load(f)
    else:
        # If the name already contains an extension, try loading directly
        if name.endswith(('.pkl', '.pt', '.pth', '.npy')):
            full_path_with_ext = os.path.join(directory, name)
            if os.path.exists(full_path_with_ext):
                print(f"Loading {os.path.splitext(name)[1]} from: {full_path_with_ext}")
                if name.endswith('.pkl'):
                    with open(full_path_with_ext, 'rb') as f:
                        return pickle.load(f)
                elif name.endswith(('.pt', '.pth')):
                    return torch.load(full_path_with_ext)
                elif name.endswith('.npy'):
                    return np.load(full_path_with_ext)

        raise FileNotFoundError(f"Object '{name}' not found in directory '{directory}' with .pkl, .pt, .pth, or .npy extensions.")