import sys
import torch
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from brats_loader import BraTSDataset
from brats_loader import collect_patient_info_from_root,get_train_transforms,get_val_transforms
from AE_BraTs import DAE
from clf_unet import UNetPP
from evaluation.evaluator import AEDetector2D, SimpleReformer2D, ClassifierUNet, OperatorBraTS, EvaluatorBraTS
from attack.attack import AttackDataBraTS, generate_attack_data_brats, save_obj, load_obj, normalize_brats
from utils import BraTSDataWrapper, brats_collate

# ---- Load BraTS-optimized models ----
detector_I = AEDetector2D(
    DAE,
    "/kaggle/input/required3/BraTS_DAE2D_I_final.pth",
    p=1,
    model_kwargs={
        'image_shape': (4, 240, 240),
        'structure': [16, "max",32, "max", "linear_bottleneck",256],
        'v_noise': 0.05,
        'activation': 'leaky_relu'
    }
)

detector_II = AEDetector2D(
    DAE,
    "/kaggle/input/required3/BraTS_DAE2D_II_final.pth",
    p=1,
    model_kwargs={
        'image_shape': (4, 240, 240),
        'structure': [16, "max",32, "max",64 ,"max",128,"max", "linear_bottleneck",128],
        'v_noise': 0.05,
        'activation': 'leaky_relu'
    }
)

classifier = ClassifierUNet(
    UNetPP,
    "/kaggle/input/required3/BraTs_UNet_actual_params.pth",
    model_kwargs={
        'in_channels': 4,
        'out_channels': 4,  # For tumor sub-regions
        'features': [64, 128, 256]
    }
)
reformer = SimpleReformer2D(
    DAE,
    "/kaggle/input/required3/BraTS_DAE2D_II_final.pth",
    device = classifier.device,
    model_kwargs={
        'image_shape': (4, 240, 240),
        'structure': [16, "max",32, "max",64 ,"max",128,"max", "linear_bottleneck",128]
    }
)

# ---- Initialize BraTS pipeline components ----
detector_dict = {
    "I": detector_I,
    "II": detector_II
}

data_root = "/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training"

patients = collect_patient_info_from_root(data_root, grade_subfolders=True)

train_patients, temp = train_test_split(patients, test_size=0.3, random_state=42)

val_patients, test_patients = train_test_split(temp, test_size=0.5, random_state=42)

train_dataset = BraTSDataset(train_patients, transform=get_train_transforms())

val_dataset = BraTSDataset(val_patients, transform=get_val_transforms())

test_dataset = BraTSDataset(test_patients, transform=get_val_transforms())

# Create DataLoaders
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=brats_collate
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=brats_collate
)

# Wrap DataLoaders
data_wrapper = BraTSDataWrapper(None, val_loader, test_loader)  # Simplified example

operator = OperatorBraTS(
    data_wrapper=data_wrapper,
    classifier=classifier,
    det_dict=detector_dict,
    reformer=reformer
)
# ---- Attack configuration ----
SAVE_DIR = '/kaggle/working/brats_attack_data/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Generate 100 random BraTS volume indices
idx = torch.randperm(len(test_dataset))[:100]
batch = next(iter(test_loader))
X_clean, targets = batch[0], batch[1]
X_clean = normalize_brats(X_clean)  # Medical image normalization

# Medical imaging-appropriate epsilon range
epsilons = [0.005, 0.025, 0.05, 0.075, 0.1]

# Generate volumetric adversarial examples
for eps in epsilons:
    attack_data = generate_attack_data_brats(
        classifier.model, 
        "fgsm", 
        eps,
        num_samples=50,
        dataset=test_dataset
    )
    save_obj(attack_data.data.cpu(), f"fgsm_{eps}_attack", directory=SAVE_DIR)
    save_obj(attack_data.labels.cpu(), f"fgsm_{eps}_labels", directory=SAVE_DIR)

# ---- Medical imaging evaluation setup ----
LOAD_DIR = '/kaggle/working/brats_attack_data/'
device = next(classifier.model.parameters()).device

def load_and_normalize_attack(attack_name):
    images = load_obj(f"{attack_name}_attack", LOAD_DIR).to(device)
    labels = load_obj(f"{attack_name}_labels", LOAD_DIR).to(device)
    return AttackDataBraTS(normalize_brats(images), labels, name=attack_name)

# Initialize evaluator with tumor segmentation metrics
initial_attack = load_and_normalize_attack("fgsm_0.005")
evaluator = EvaluatorBraTS(operator, initial_attack)

# Generate clinical performance visualization
evaluator.plot_epsilon_sweep(
    epsilons=epsilons,
    drop_rate={"I": 0.1, "II": 0.1},
    graph_name="brats_fgsm_epsilon_analysis",
    attack_type="fgsm"
)

