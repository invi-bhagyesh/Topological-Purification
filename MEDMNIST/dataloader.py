import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

def download_npz_if_needed(data_flag):
    info = INFO[data_flag]
    filename = f"{data_flag}.npz"
    url = f"https://zenodo.org/records/10519652/files/{filename}?download=1"
    local_path = os.path.expanduser(f"~/.medmnist/{filename}")

    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Zenodo...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        os.system(f"wget -O {local_path} {url}")
        print("Download complete.")

def load_medmnist(data_flag='pathmnist', batch_size=64, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Check and manually download if needed
    if download:
        download_npz_if_needed(data_flag)

    # Get the number of channels from the info dictionary
    n_channels = info.get('n_channels', 3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5]*n_channels, std=[.5]*n_channels)
    ])

    train_dataset = DataClass(split='train', transform=transform, download=False)
    val_dataset = DataClass(split='val', transform=transform, download=False)
    test_dataset = DataClass(split='test', transform=transform, download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, info

if __name__ == '__main__':
    loaders = load_medmnist('pathmnist')
    train_loader, _, _, info = loaders
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        break
