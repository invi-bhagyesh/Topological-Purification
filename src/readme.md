# Topological Purification 

This directory contains the main training and testing scripts for the Topological Purification project with comprehensive command line argument support.

## Files

- `train.py` - Training script for DAE models
- `test.py` - Testing and evaluation script with adversarial attacks
- `run_examples.py` - Example usage demonstrations
- `utils.py` - Utility functions and data wrappers

## Quick Start

### Training

```bash
# Basic training with defaults
python train.py

# Train with custom parameters
python train.py --epochs 20 --batch_size 8 --learning_rate 0.0005

# Train only Model I
python train.py --model_type I --epochs 15

# Train with custom data path
python train.py --data_root /path/to/your/brats/data --output_dir ./my_models/
```

### Testing

```bash
# Basic testing with defaults
python test.py

# Test with custom attack parameters
python test.py --attack_type pgd --epsilons 0.01,0.02,0.03

# Test with custom model paths
python test.py --detector_I_path ./my_models/BraTS_DAE2D_I_final.pth

# Test with different data split
python test.py --test_size 0.2 --val_size 0.3
```

## Command Line Arguments

### Training Script (`train.py`)

#### Data Parameters
- `--data_root` (str): Path to BraTS data root directory
  - Default: `/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training`
- `--batch_size` (int): Batch size for training
  - Default: 4
- `--num_workers` (int): Number of workers for data loading
  - Default: 4

#### Training Parameters
- `--epochs` (int): Number of training epochs
  - Default: 8
- `--learning_rate` (float): Learning rate for optimizer
  - Default: 0.001
- `--reg_strength` (float): Weight decay/regularization strength
  - Default: 1e-9
- `--activation` (str): Activation function
  - Choices: ['relu', 'leaky_relu', 'tanh']
  - Default: 'relu'

#### Model Parameters
- `--input_shape` (str): Input shape as comma-separated values (C,H,W)
  - Default: '4,240,240'
- `--v_noise` (float): Noise level for denoising autoencoder
  - Default: 0.1
- `--model_type` (str): Which DAE model to train
  - Choices: ['I', 'II', 'both']
  - Default: 'both'

#### Structure Parameters
- `--structure_I` (str): Structure for Model I as comma-separated values
  - Format: numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck
  - Default: '32,max,64,max,128,max,linear_bottleneck,2048'
- `--structure_II` (str): Structure for Model II as comma-separated values
  - Format: numbers for channels, "max" for maxpool
  - Default: '64,max,128,max,256,max,512'

#### Output Parameters
- `--output_dir` (str): Directory to save trained models
  - Default: './defensive_models/'
- `--use_wandb` (flag): Enable wandb logging
  - Default: False

### Testing Script (`test.py`)

#### Data Parameters
- `--data_root` (str): Path to BraTS data root directory
  - Default: `/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training`
- `--batch_size` (int): Batch size for testing
  - Default: 4
- `--num_workers` (int): Number of workers for data loading
  - Default: 4
- `--test_size` (float): Fraction of data to use for testing
  - Default: 0.3
- `--val_size` (float): Fraction of remaining data to use for validation
  - Default: 0.5
- `--random_state` (int): Random seed for data splitting
  - Default: 42

#### Model Paths
- `--detector_I_path` (str): Path to DAE Model I weights
  - Default: `/kaggle/input/required3/BraTS_DAE2D_I_final.pth`
- `--detector_II_path` (str): Path to DAE Model II weights
  - Default: `/kaggle/input/required3/BraTS_DAE2D_II_final.pth`
- `--classifier_path` (str): Path to UNet classifier weights
  - Default: `/kaggle/input/required3/BraTs_UNet_actual_params.pth`

#### Model Parameters
- `--input_shape` (str): Input shape as comma-separated values (C,H,W)
  - Default: '4,240,240'
- `--v_noise` (float): Noise level for denoising autoencoder
  - Default: 0.05
- `--activation` (str): Activation function
  - Choices: ['relu', 'leaky_relu', 'tanh']
  - Default: 'leaky_relu'

#### Structure Parameters
- `--structure_I` (str): Structure for Model I as comma-separated values
  - Format: numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck
  - Default: '16,max,32,max,linear_bottleneck,256'
- `--structure_II` (str): Structure for Model II as comma-separated values
  - Format: numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck
  - Default: '16,max,32,max,64,max,128,max,linear_bottleneck,128'
- `--reformer_structure` (str): Structure for reformer model as comma-separated values
  - Format: numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck
  - Default: '16,max,32,max,64,max,128,max,linear_bottleneck,128'

#### Attack Parameters
- `--attack_type` (str): Type of adversarial attack
  - Choices: ['fgsm', 'pgd', 'cw']
  - Default: 'fgsm'
- `--epsilons` (str): Epsilon values for attacks as comma-separated list
  - Default: '0.005,0.025,0.05,0.075,0.1'
- `--num_attack_samples` (int): Number of samples to generate for each attack
  - Default: 50

#### Output Parameters
- `--save_dir` (str): Directory to save attack data
  - Default: '/kaggle/working/brats_attack_data/'
- `--load_dir` (str): Directory to load attack data from
  - Default: '/kaggle/working/brats_attack_data/'
- `--drop_rate` (str): Drop rates for detectors as comma-separated list (I,II)
  - Default: '0.1,0.1'
- `--graph_name` (str): Name for the epsilon analysis graph
  - Default: 'brats_fgsm_epsilon_analysis'

## Examples

### Training Examples

1. **Basic training with all defaults:**
   ```bash
   python train.py
   ```

2. **Train only Model I with custom hyperparameters:**
   ```bash
   python train.py --epochs 20 --batch_size 8 --learning_rate 0.0005 --model_type I
   ```

3. **Train with custom data path and output directory:**
   ```bash
   python train.py --data_root /path/to/your/brats/data --output_dir ./my_models/
   ```

4. **Train Model II with different activation and noise:**
   ```bash
   python train.py --activation leaky_relu --v_noise 0.15 --model_type II
   ```

5. **Train with custom model architectures:**
   ```bash
   python train.py --structure_I '64,max,128,max,linear_bottleneck,1024' --structure_II '32,max,64,max,128,max,256' --model_type both
   ```

6. **Train with simplified structure:**
   ```bash
   python train.py --structure_I '16,max,32,max,linear_bottleneck,512' --model_type I --epochs 10
   ```

7. **Train with wandb logging:**
   ```bash
   python train.py --use_wandb --epochs 50 --batch_size 16
   ```

### Testing Examples

1. **Basic testing with all defaults:**
   ```bash
   python test.py
   ```

2. **Test with PGD attack and custom epsilon values:**
   ```bash
   python test.py --attack_type pgd --epsilons 0.01,0.02,0.03 --num_attack_samples 100
   ```

3. **Test with custom model paths:**
   ```bash
   python test.py --detector_I_path ./my_models/BraTS_DAE2D_I_final.pth --detector_II_path ./my_models/BraTS_DAE2D_II_final.pth
   ```

4. **Test with custom data split ratios:**
   ```bash
   python test.py --test_size 0.2 --val_size 0.3 --random_state 123
   ```

5. **Test with custom model architectures:**
   ```bash
   python test.py --structure_I '32,max,64,max,linear_bottleneck,512' --structure_II '16,max,32,max,64,max,128,max,linear_bottleneck,256' --reformer_structure '16,max,32,max,64,max,128,max,linear_bottleneck,256'
   ```

6. **Test with custom directories for attack data:**
   ```bash
   python test.py --save_dir ./attack_results/ --load_dir ./attack_results/ --graph_name my_analysis
   ```

## Getting Help

To see all available options for each script:

```bash
python train.py --help
python test.py --help
```

## Running Examples

To see all example commands:

```bash
python run_examples.py
```

This will display various example commands without actually executing them.

## Notes

- All paths should be absolute or relative to the current working directory
- The scripts will create output directories if they don't exist
- GPU will be used automatically if available, otherwise CPU
- The default parameters are optimized for the BraTS dataset
- Make sure all required dependencies are installed before running the scripts
