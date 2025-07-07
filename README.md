# CLI Usage Documentation

This project provides command-line tools for training and evaluating Denoising Autoencoder (DAE) models on medical imaging datasets.

## Files

- `train.py` — Train DAE models for BraTS and MedMNIST data.
- `test.py` — Evaluate trained models and run adversarial attacks on BraTS data.

---

## Quick Start

### Training

```bash
# Basic training with default settings
python train.py

# Train with custom parameters
python train.py --epochs 20 --batch_size 8 --learning_rate 0.0005

# Train only Model I
python train.py --model_type I --epochs 15

# Train with a custom data path
python train.py --data_root /path/to/brats/data --output_dir ./my_models/
```

### Testing

```bash
# Basic testing with default settings
python test.py

# Test with custom attack parameters
python test.py --attack_type pgd --epsilons 0.01,0.02,0.03

# Test with custom model paths
python test.py --detector_I_path ./my_models/BraTS_DAE2D_I_final.pth

# Test with different data split
python test.py --test_size 0.2 --val_size 0.3
```

---

## Command Line Arguments

### `train.py`

**Data Parameters**
- `--data_root` (str): Path to BraTS data root directory (default: `/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training`)
- `--medmnist_root` (str): Path to MedMNIST data root directory (default: `./data/medmnist`)
- `--batch_size` (int): Batch size for training (default: 4)
- `--num_workers` (int): Number of workers for data loading (default: 4)

**Training Parameters**
- `--epochs` (int): Number of training epochs (default: 8)
- `--learning_rate` (float): Learning rate for optimizer (default: 0.001)
- `--reg_strength` (float): Weight decay/regularization strength (default: 1e-9)
- `--activation` (str): Activation function (`relu`, `leaky_relu`, `tanh`, `sigmoid`, `elu`, `gelu`; default: `relu`)

**Model Parameters**
- `--input_shape` (str): Input shape as comma-separated values (C,H,W) (default: `4,240,240`)
- `--v_noise` (float): Noise level for denoising autoencoder (default: 0.1)
- `--task_type` (str): Task type (`segmentation`, `multiclass`, `binary`, `reconstruction`; default: `reconstruction`)
- `--dataset` (str): Dataset name (`brats`, `pathmnist`, etc.; default: `brats`)
- `--num_classes` (int): Number of classes for classification/segmentation tasks (default: 4)
- `--loss_type` (str): Loss type (`reconstruction`, `simclr`; default: `reconstruction`)
- `--projection_dim` (int): Projection dimension for SimCLR (default: 128)
- `--temperature` (float): Temperature for SimCLR loss (default: 0.1)
- `--model_type` (str): Which DAE model to train (`I`, `II`, `both`; default: `both`)

**Structure Parameters**
- `--structure_I` (str): Structure for Model I as comma-separated values (default: `32,max,64,max,128,max,linear_bottleneck,2048`)
- `--structure_II` (str): Structure for Model II as comma-separated values (default: `64,max,128,max,256,max,512`)

**Output Parameters**
- `--output_dir` (str): Directory to save trained models (default: `./defensive_models/`)
- `--use_wandb` (flag): Enable wandb logging

---

### `test.py`

**Data Parameters**
- `--data_root` (str): Path to BraTS data root directory (default: `/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training`)
- `--batch_size` (int): Batch size for testing (default: 4)
- `--num_workers` (int): Number of workers for data loading (default: 4)
- `--test_size` (float): Fraction of data to use for testing (default: 0.3)
- `--val_size` (float): Fraction of remaining data to use for validation (default: 0.5)
- `--random_state` (int): Random seed for data splitting (default: 42)

**Model Paths**
- `--detector_I_path` (str): Path to DAE Model I weights (default: `/kaggle/input/required3/BraTS_DAE2D_I_final.pth`)
- `--detector_II_path` (str): Path to DAE Model II weights (default: `/kaggle/input/required3/BraTS_DAE2D_II_final.pth`)
- `--classifier_path` (str): Path to UNet classifier weights (default: `/kaggle/input/required3/BraTs_UNet_actual_params.pth`)

**Model Parameters**
- `--input_shape` (str): Input shape as comma-separated values (C,H,W) (default: `4,240,240`)
- `--v_noise` (float): Noise level for denoising autoencoder (default: 0.05)
- `--activation` (str): Activation function (`relu`, `leaky_relu`, `tanh`; default: `leaky_relu`)

**Structure Parameters**
- `--structure_I` (str): Structure for Model I as comma-separated values (default: `16,max,32,max,linear_bottleneck,256`)
- `--structure_II` (str): Structure for Model II as comma-separated values (default: `16,max,32,max,64,max,128,max,linear_bottleneck,128`)
- `--reformer_structure` (str): Structure for reformer model as comma-separated values (default: `16,max,32,max,64,max,128,max,linear_bottleneck,128`)

**Attack Parameters**
- `--attack_type` (str): Type of adversarial attack (`fgsm`, `pgd`, `cw`; default: `fgsm`)
- `--epsilons` (str): Epsilon values for attacks as comma-separated list (default: `0.005,0.025,0.05,0.075,0.1`)
- `--num_attack_samples` (int): Number of samples to generate for each attack (default: 50)

**Output Parameters**
- `--save_dir` (str): Directory to save attack data (default: `/kaggle/working/brats_attack_data/`)
- `--load_dir` (str): Directory to load attack data from (default: `/kaggle/working/brats_attack_data/`)
- `--drop_rate` (str): Drop rates for detectors as comma-separated list (I,II; default: `0.1,0.1`)
- `--graph_name` (str): Name for the epsilon analysis graph (default: `brats_fgsm_epsilon_analysis`)

---

## Getting Help

To see all available options for each script:

```bash
python train.py --help
python test.py --help
```

---

## Notes

- All paths can be absolute or relative to the current working directory.
- The scripts will create output directories if they don't exist.
- GPU will be used automatically if available, otherwise CPU.
- Default parameters are optimized for the BraTS dataset.
- Make sure all required dependencies are installed before running the scripts.
