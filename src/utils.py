import pickle
import os
import torch


def prepare_data(dataset, idx):
    """
    Extracts image and label data at specific indices from the test set.

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
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, directory='./attack_data/'):
    if name.endswith(".pkl"):
        name = name[:-4]
    with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
