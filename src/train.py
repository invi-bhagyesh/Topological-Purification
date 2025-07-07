import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
import argparse
from model.DAE import DAE
from data.dataloader import get_brats_dataloaders
from data.medmnist_loader import get_medmnist_dataloaders 

# Dataset configurations
DATASET_CONFIGS = {
    'brats': {
        'task_type': 'segmentation',
        'num_classes': 4,
        'input_shape': (4, 240, 240),
        'is_medmnist': False
    },
    'pathmnist': {
        'task_type': 'multiclass',
        'num_classes': 9,
        'input_shape': (3, 28, 28),
        'is_medmnist': True
    },
    'dermamnist': {
        'task_type': 'multiclass',
        'num_classes': 7,
        'input_shape': (3, 28, 28),
        'is_medmnist': True
    },
    'bloodmnist': {
        'task_type': 'binary',
        'num_classes': 2,
        'input_shape': (3, 28, 28),
        'is_medmnist': True
    },
    'chestmnist': {
        'task_type': 'binary',
        'num_classes': 2,
        'input_shape': (1, 28, 28),
        'is_medmnist': True
    },
    'breastmnist': {
        'task_type': 'binary',
        'num_classes': 2,
        'input_shape': (1, 28, 28),
        'is_medmnist': True
    },
    'pneumoniamnist': {
        'task_type': 'binary',
        'num_classes': 2,
        'input_shape': (1, 28, 28),
        'is_medmnist': True
    }
}

#train.py


def parse_args():
    parser = argparse.ArgumentParser(description='Train DAE models for BraTS and MedMNIST data')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, 
                       default='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
                       help='Path to BraTS data root directory')
    parser.add_argument('--medmnist_root', type=str, default='./data/medmnist',
                       help='Path to MedMNIST data root directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=8, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Learning rate for optimizer')
    parser.add_argument('--reg_strength', type=float, default=1e-9, 
                       help='Weight decay/regularization strength')
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'elu', 'gelu'], 
                       help='Activation function')
    
    # Model parameters
    parser.add_argument('--input_shape', type=str, default='4,240,240', 
                       help='Input shape as comma-separated values (C,H,W)')
    parser.add_argument('--v_noise', type=float, default=0.1, 
                       help='Noise level for denoising autoencoder')
    
    # Enhanced DAE parameters
    parser.add_argument('--task_type', type=str, default='reconstruction', 
                       choices=['segmentation', 'multiclass', 'binary', 'reconstruction'], 
                       help='Task type for the model')
    parser.add_argument('--dataset', type=str, default='brats', 
                       choices=['brats', 'pathmnist', 'dermamnist', 'bloodmnist', 'chestmnist', 'breastmnist', 'pneumoniamnist'],
                       help='Dataset name')
    parser.add_argument('--num_classes', type=int, default=4, 
                       help='Number of classes for classification/segmentation tasks')
    parser.add_argument('--loss_type', type=str, default='reconstruction', 
                       choices=['reconstruction', 'simclr'], 
                       help='Loss type to use')
    parser.add_argument('--projection_dim', type=int, default=128, 
                       help='Projection dimension for SimCLR')
    parser.add_argument('--temperature', type=float, default=0.1, 
                       help='Temperature for SimCLR loss')
    
    # Model architecture parameters
    parser.add_argument('--model_type', type=str, default='both', 
                       choices=['I', 'II', 'both'], 
                       help='Which DAE model to train (I, II, or both)')
    
    # Structure parameters
    parser.add_argument('--structure_I', type=str, 
                       default='32,max,64,max,128,max,linear_bottleneck,2048',
                       help='Structure for Model I as comma-separated values')
    parser.add_argument('--structure_II', type=str, 
                       default='64,max,128,max,256,max,512',
                       help='Structure for Model II as comma-separated values')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./defensive_models/', 
                       help='Directory to save trained models')
    parser.add_argument('--use_wandb', action='store_true', 
                       help='Enable wandb logging')
    
    args, unknown = parser.parse_known_args()  # Accept unknown args like -f
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_structure(structure_str):
    """Parse structure string into list of integers and strings"""
    structure = []
    for item in structure_str.split(','):
        item = item.strip()
        if item == 'max':
            structure.append('max')
        elif item == 'linear_bottleneck':
            structure.append('linear_bottleneck')
        else:
            try:
                structure.append(int(item))
            except ValueError:
                raise ValueError(f"Invalid structure item: {item}. Must be integer, 'max', or 'linear_bottleneck'")
    return structure


def get_dataloaders(args):
    """Get appropriate dataloaders based on dataset type"""
    dataset_config = DATASET_CONFIGS[args.dataset]
    
    if dataset_config['is_medmnist']:
        print(f"Loading MedMNIST dataset: {args.dataset}")
        return get_medmnist_dataloaders(
            dataset_name=args.dataset,
            data_root=args.medmnist_root,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, os.cpu_count())
        )
    else:
        print(f"Loading BraTS dataset")
        return get_brats_dataloaders(
            train_data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, os.cpu_count())
        )

def train_autoencoder(model, train_loader, val_loader, archive_name, args):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.reg_strength)
    
    best_val_loss = float('inf')
    dataset_config = DATASET_CONFIGS[args.dataset]

    print(f"\n--- Starting training for {archive_name} ---")
    print(f"Dataset: {args.dataset}")
    print(f"Task type: {model.task_type}")
    print(f"Loss type: {model.loss_type}")
    print(f"Device: {device}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            if dataset_config['is_medmnist']:
                # MedMNIST format: (images, labels)
                imgs, labels = batch_data
                imgs = imgs.to(device)
                labels = labels.to(device)
            else:
                # BraTS format: dict with 'image' key
                imgs = batch_data['image']
                imgs = imgs.to(device)
                labels = batch_data.get('mask', None)
                if labels is not None:
                    labels = labels.to(device)

            if epoch == 0 and batch_idx == 0:
                print(f"Input batch shape: {imgs.shape}")
                assert imgs.dim() == 4, f"Expected 4D tensor (B,C,H,W), got {imgs.shape}"

            # Prepare targets based on task type
            if model.task_type == 'segmentation':
                # For segmentation, we need segmentation masks
                if labels is not None:
                    targets = labels
                else:
                    # If no segmentation masks available, use original images for reconstruction
                    print("Warning: No segmentation masks found, using reconstruction targets")
                    targets = imgs
            elif model.task_type in ['multiclass', 'binary']:
                # For classification tasks
                if dataset_config['is_medmnist']:
                    # MedMNIST provides labels directly
                    targets = {
                        'images': imgs,
                        'labels': labels
                    }
                else:
                    # BraTS case
                    targets = {
                        'images': imgs,
                        'labels': labels if labels is not None else torch.zeros(imgs.shape[0], dtype=torch.long).to(device)
                    }
            else:
                # Pure reconstruction task
                targets = imgs

            optimizer.zero_grad()
            
            # Forward pass
            if model.loss_type == 'simclr':
                # For SimCLR, we need augmented pairs
                # Create augmented versions with different noise levels
                aug1 = imgs + args.v_noise * torch.randn_like(imgs)
                aug2 = imgs + (args.v_noise * 1.5) * torch.randn_like(imgs)
                
                # Clamp to valid range
                aug1 = torch.clamp(aug1, 0.0, 1.0)
                aug2 = torch.clamp(aug2, 0.0, 1.0)
                
                # Combine augmented views
                combined_input = torch.cat([aug1, aug2], dim=0)
                outputs = model(combined_input)
                loss = model.get_loss(outputs)
            else:
                # Regular reconstruction or classification
                outputs = model(imgs)
                loss = model.get_loss(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(f"Epoch [{epoch + 1}/{args.epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                if dataset_config['is_medmnist']:
                    # MedMNIST format
                    imgs, labels = batch_data
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                else:
                    # BraTS format
                    imgs = batch_data['image'].to(device)
                    labels = batch_data.get('mask', None)
                    if labels is not None:
                        labels = labels.to(device)
                
                # Prepare validation targets
                if model.task_type == 'segmentation':
                    if labels is not None:
                        targets = labels
                    else:
                        targets = imgs
                elif model.task_type in ['multiclass', 'binary']:
                    if dataset_config['is_medmnist']:
                        targets = {
                            'images': imgs,
                            'labels': labels
                        }
                    else:
                        targets = {
                            'images': imgs,
                            'labels': labels if labels is not None else torch.zeros(imgs.shape[0], dtype=torch.long).to(device)
                        }
                else:
                    targets = imgs

                if model.loss_type == 'simclr':
                    # For SimCLR validation
                    aug1 = imgs + args.v_noise * torch.randn_like(imgs)
                    aug2 = imgs + (args.v_noise * 1.5) * torch.randn_like(imgs)
                    aug1 = torch.clamp(aug1, 0.0, 1.0)
                    aug2 = torch.clamp(aug2, 0.0, 1.0)
                    combined_input = torch.cat([aug1, aug2], dim=0)
                    outputs = model(combined_input)
                    loss = model.get_loss(outputs)
                else:
                    outputs = model(imgs)
                    loss = model.get_loss(outputs, targets)
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{args.output_dir}/{archive_name}_best.pth")
            print(f"  --> Best model for {archive_name} saved with Val Loss: {best_val_loss:.4f}")

        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                f"{archive_name}_train_loss": avg_loss,
                f"{archive_name}_val_loss": avg_val_loss,
                "epoch": epoch + 1
            })

    torch.save(model.state_dict(), f"{args.output_dir}/{archive_name}_final.pth")
    print(f"Final model for {archive_name} saved.")

def create_model(input_shape, structure, args, model_name):
    """Create a DAE model with specified parameters."""
    print(f"\nCreating DAE Model {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Task type: {args.task_type}")
    print(f"Loss type: {args.loss_type}")
    print(f"Structure: {structure}")
    
    # Determine number of classes based on task type
    num_classes = None
    if args.task_type in ['segmentation', 'multiclass', 'binary']:
        num_classes = args.num_classes
    
    model = DAE(
        image_shape=input_shape,
        structure=structure,
        task_type=args.task_type,
        dataset=args.dataset,
        num_classes=num_classes,
        v_noise=args.v_noise,
        activation=args.activation,
        reg_strength=args.reg_strength,
        loss_type=args.loss_type,
        projection_dim=args.projection_dim,
        temperature=args.temperature
    )
    return model

def adjust_structure_for_input_size(structure, input_shape):
    """Adjust structure based on input size for better performance"""
    _, height, width = input_shape
    
    # For small images (28x28), use smaller structures
    if height <= 32 and width <= 32:
        # Reduce the complexity for small images
        adjusted_structure = []
        for item in structure:
            if isinstance(item, int):
                # Scale down feature maps for small images
                adjusted_structure.append(min(item, 256))
            else:
                adjusted_structure.append(item)
        return adjusted_structure
    
    return structure

def main():
    args = parse_args()
    
    # Get dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(DATASET_CONFIGS.keys())}")
    
    dataset_config = DATASET_CONFIGS[args.dataset]
    
    # Override task type and num_classes based on dataset if not explicitly set
    if args.task_type == 'reconstruction' and dataset_config['task_type'] != 'reconstruction':
        args.task_type = dataset_config['task_type']
        print(f"Auto-setting task_type to '{args.task_type}' for dataset '{args.dataset}'")
    
    if args.num_classes == 4 and dataset_config['num_classes'] != 4:
        args.num_classes = dataset_config['num_classes']
        print(f"Auto-setting num_classes to {args.num_classes} for dataset '{args.dataset}'")
    
    # Parse input shape and structures
    if args.input_shape == '4,240,240':
        # Use dataset-specific input shape
        input_shape = dataset_config['input_shape']
        print(f"Using dataset-specific input shape: {input_shape}")
    else:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    
    structure_I = parse_structure(args.structure_I)
    structure_II = parse_structure(args.structure_II)
    
    # Adjust structures for smaller input sizes
    structure_I = adjust_structure_for_input_size(structure_I, input_shape)
    structure_II = adjust_structure_for_input_size(structure_II, input_shape)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=f"{args.dataset}-dae-training",
            config=vars(args),
            name=f"DAE_{args.dataset}_{args.task_type}_{args.loss_type}_{args.model_type}"
        )
    
    # Load appropriate dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args)
    print(f"DataLoaders ready for {args.dataset}.")

    print(f"\nTraining Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Input shape: {input_shape}")
    print(f"Task type: {args.task_type}")
    print(f"Loss type: {args.loss_type}")
    if args.task_type in ['segmentation', 'multiclass', 'binary']:
        print(f"Number of classes: {args.num_classes}")
    print(f"Model I structure: {structure_I}")
    print(f"Model II structure: {structure_II}")
    
    # Create archive name suffix based on task and loss type
    suffix = f"{args.task_type}_{args.loss_type}"
    if args.loss_type == 'simclr':
        suffix += f"_proj{args.projection_dim}_temp{args.temperature}"
    
    if args.model_type in ['I', 'both']:
        AE_I = create_model(input_shape, structure_I, args, "I")
        archive_name = f"{args.dataset.upper()}_DAE2D_I_{suffix}"
        train_autoencoder(AE_I, train_loader, val_loader, archive_name, args)

    if args.model_type in ['II', 'both']:
        AE_II = create_model(input_shape, structure_II, args, "II")
        archive_name = f"{args.dataset.upper()}_DAE2D_II_{suffix}"
        train_autoencoder(AE_II, train_loader, val_loader, archive_name, args)

    print("\nTraining complete!")
    print(f"Test Loader has {len(test_loader.dataset)} samples.")
    
    # Print summary of trained models
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Task Type: {args.task_type}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Models Trained: {args.model_type}")
    print(f"Input Shape: {input_shape}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Saved to: {args.output_dir}")
    print("="*60)
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
