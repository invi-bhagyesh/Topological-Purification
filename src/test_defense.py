# test_defense.py
import sys
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# # Import modules - simplified import structure
# try:
#     # Try importing from src package first
#     from src.worker import AEDetector, SimpleReformer, IdReformer, Classifier, Operator, Evaluator, AttackData
#     from src.mnist_loader import BraTSDataset, get_train_transforms, get_val_transforms, get_brats_dataloaders, MNIST
#     from src.utils import load_obj
#     from src.model.DAE_model import DAE  # Updated to match your actual class name
#     from src.model.clf_unet import UNet as BraTSUnetClassifier
#     from src.model.clf_cnn import CNNClassifier
# except ImportError:
#     # Fallback to direct imports
#     from worker import AEDetector, SimpleReformer, IdReformer, Classifier, Operator, Evaluator, AttackData
#     from mnist_loader import BraTSDataset, get_train_transforms, get_val_transforms, get_brats_dataloaders, MNIST
#     from utils import load_obj
#     from DAE_model import DAE  # Updated to match your actual class name
#     from clf_unet import UNet as BraTSUnetClassifier
#     from clf_cnn import CNNClassifier

# --- Configuration ---
DATASET_NAME = "brats"  # Set to "brats" or "mnist"

# --- Common BraTS Configuration ---
BRATS_CHANNELS = 4  # T1, T1ce, T2, FLAIR
BRATS_IMG_SIZE = 240  # BraTS images are typically 240x240

# --- Initialize variables ---
dataset = None
operator = None
evaluator = None
test_attack = None

# --- Constants for paths ---
BASE_DATA_DIR = '/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/'
MODEL_SAVE_DIR = '/kaggle/input/3d-unet/pytorch/default/1/'
ADVERSARIAL_EXAMPLES_DIR = './adversial_example/'

# ---- Load models and data based on DATASET_NAME ----
if DATASET_NAME == "mnist":
    print("Setting up for MNIST dataset...")
    
    # ---- Load models ----
    mnist_ae_path_I = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "MNIST_I.pth")
    mnist_ae_path_II = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "MNIST_II.pth")
    mnist_classifier_path = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "example_classifier.pth")

    # Updated model_kwargs to match DAE class parameters
    mnist_dae_kwargs = {
        'image_shape': (1, 28, 28),  # (channels, height, width)
        'structure': [16, "max", 32, "linear_bottleneck", 128],  # Added bottleneck size
        'v_noise': 0.1,
        'activation': 'relu',  # String format as handled by DAE class
        'reg_strength': 1e-4
    }

    detector_I = AEDetector(DAE, mnist_ae_path_I, p=2, model_kwargs=mnist_dae_kwargs)
    detector_II = AEDetector(DAE, mnist_ae_path_II, p=1, model_kwargs=mnist_dae_kwargs)
    reformer = SimpleReformer(DAE, mnist_ae_path_I, model_kwargs=mnist_dae_kwargs)
    id_reformer = IdReformer()
    
    classifier = Classifier(CNNClassifier, mnist_classifier_path,
                           model_kwargs={'params': [32, 32, 64, 64, 200, 200], 
                                       'in_channels': 1, 'num_classes': 10})

    # ---- Compose detector dictionary ----
    detector_dict = {"I": detector_I, "II": detector_II}

    # ---- Load MNIST data ----
    dataset = MNIST()
    operator = Operator(dataset, classifier, detector_dict, reformer, task_type="classification")

    # ---- Load adversarial example indices and labels ----
    idx_file_path = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "example_idx")
    idx = load_obj(idx_file_path)
    mnist_test_data_len = len(dataset.test_data)
    idx = [i for i in idx if i < mnist_test_data_len]

    Y_labels = torch.tensor([dataset.test_data[i][1] for i in idx], dtype=torch.long)

    # ---- Load adversarial examples ----
    examples_np_path = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "example_carlini_0.0")
    examples_np = load_obj(examples_np_path)

    if len(examples_np) != len(idx):
        print(f"Warning: Number of examples ({len(examples_np)}) != indices ({len(idx)}). Adjusting...")
        examples_np = examples_np[:len(idx)]

    examples = torch.tensor(examples_np, dtype=torch.float32)
    if examples.ndim == 4 and examples.shape[-1] == 1:
        examples = examples.permute(0, 3, 1, 2)
    elif examples.ndim == 3 and examples.shape[-1] == 1:
        examples = examples.permute(2, 0, 1).unsqueeze(0)
    elif examples.ndim == 2:
        examples = examples.unsqueeze(0).unsqueeze(0)

    test_attack = AttackData(examples, Y_labels, name="Carlini L2 0.0")
    evaluator = Evaluator(operator, test_attack)

    # ---- Plot performance ----
    mnist_drop_rate = {"I": 0.1, "II": 0.1}
    evaluator.plot_various_confidences(
        graph_name="defense_performance_mnist",
        drop_rate=mnist_drop_rate,
        idx_file="example_idx",
        get_attack_data_name=lambda c: f"example_carlini_{c}",
        data_dir=ADVERSARIAL_EXAMPLES_DIR
    )

elif DATASET_NAME == "brats":
    print("Setting up for BraTS dataset...")
    
    # --- Paths for BraTS models ---
    brats_dae_model_path = os.path.join(MODEL_SAVE_DIR, "BraTS_DAE2D_I_best.pth")
    brats_segmenter_model_path = os.path.join(MODEL_SAVE_DIR, "brats_segmentation_model_local_split.pth")

    # ---- Load BraTS models ----
    # Updated model_kwargs to match DAE class parameters
    brats_dae_kwargs = {
        'image_shape': (BRATS_CHANNELS, BRATS_IMG_SIZE, BRATS_IMG_SIZE),  # (4, 240, 240)
        'structure': [64, "max", 128, "max", 256, "linear_bottleneck", 512],  # Added bottleneck size
        'v_noise': 0.1,
        'activation': 'relu',
        'reg_strength': 1e-4
    }

    detector_I = AEDetector(DAE, brats_dae_model_path, p=2, model_kwargs=brats_dae_kwargs)
    detector_II = AEDetector(DAE, brats_dae_model_path, p=1, model_kwargs=brats_dae_kwargs)
    reformer = SimpleReformer(DAE, brats_dae_model_path, model_kwargs=brats_dae_kwargs)
    id_reformer = IdReformer()

    # BraTS classifier (UNet for segmentation)
    classifier = Classifier(BraTSUnetClassifier, brats_segmenter_model_path,
                           model_kwargs={'in_channels': BRATS_CHANNELS, 'num_classes': 3})

    detector_dict = {"I": detector_I, "II": detector_II}

    # ---- Load BraTS data ----
    print("Loading BraTS DataLoaders...")
    train_loader, val_loader, local_test_loader = get_brats_dataloaders(
        train_data_root=os.path.join(BASE_DATA_DIR, 'MICCAI_BraTS_2019_Data_Training'),
        batch_size=1,
        train_ratio=0.7,
        val_ratio=0.15,
        local_test_ratio=0.15,
        num_workers=os.cpu_count() // 2
    )
    print(f"BraTS DataLoaders loaded. Local test set has {len(local_test_loader.dataset)} slices.")

    # Create operator with data loaders
    operator = Operator(data_loaders=(train_loader, val_loader, local_test_loader),
                       classifier=classifier,
                       det_dict=detector_dict,
                       reformer=reformer)

    # ---- Load or generate adversarial examples for BraTS ----
    print("Loading BraTS adversarial examples...")
    try:
        adv_examples_path = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "brats_carlini_adv_examples_0.0.pt")
        adv_labels_path = os.path.join(ADVERSARIAL_EXAMPLES_DIR, "brats_carlini_adv_labels_0.0.pt")

        brats_adv_images = torch.load(adv_examples_path)
        brats_adv_labels = torch.load(adv_labels_path)

        if brats_adv_images.ndim == 3:
            brats_adv_images = brats_adv_images.unsqueeze(0)
        
        expected_shape = (BRATS_CHANNELS, BRATS_IMG_SIZE, BRATS_IMG_SIZE)
        if brats_adv_images.shape[1:] != expected_shape:
            raise ValueError(f"Adversarial images shape {brats_adv_images.shape} != expected (N, {expected_shape})")

        print(f"Loaded {len(brats_adv_images)} adversarial BraTS examples.")

    except FileNotFoundError:
        print("Adversarial BraTS examples not found. Using clean test images as placeholder.")
        temp_adv_images_list = []
        temp_adv_labels_list = []
        
        for i, batch in enumerate(local_test_loader):
            if i >= 10:  # Use small subset for testing
                break
            temp_adv_images_list.append(batch['image'].squeeze(0))
            # Convert mask to binary label (tumor present/absent)
            temp_adv_labels_list.append((batch['mask'].sum(dim=[1,2,3]) > 0).long().squeeze(0))

        brats_adv_images = torch.stack(temp_adv_images_list)
        brats_adv_labels = torch.stack(temp_adv_labels_list)
        print(f"Using {len(brats_adv_images)} clean images as placeholder adversarial examples.")

    test_attack = AttackData(brats_adv_images, brats_adv_labels, 
                            name="BraTS Adversarial Examples", 
                            directory=ADVERSARIAL_EXAMPLES_DIR)

    evaluator = Evaluator(operator, test_attack)

    # ---- Plot performance ----
    brats_drop_rate = {"I": 0.05, "II": 0.05}
    evaluator.plot_various_confidences(
        graph_name="defense_performance_brats",
        drop_rate=brats_drop_rate,
        idx_file="dummy_idx_for_brats",
        confs=[0.0, 0.1, 0.2, 0.3, 0.4],
        get_attack_data_name=lambda c: f"brats_carlini_adv_examples_{c}.pt",
        data_dir=ADVERSARIAL_EXAMPLES_DIR
    )

else:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}. Choose 'mnist' or 'brats'.")

print("\nDefense testing script finished.")