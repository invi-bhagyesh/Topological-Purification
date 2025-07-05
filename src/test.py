import sys
import torch
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data.dataloader import BraTSDataset, collect_patient_info_from_root, get_train_transforms, get_val_transforms
from model.DAE import DAE
from model.UNetPP import UNetPP
from evaluation.evaluator import AEDetector2D, SimpleReformer2D, ClassifierUNet, OperatorBraTS, EvaluatorBraTS
from attack.attack import AttackDataBraTS, generate_attack_data_brats, save_obj, load_obj, normalize_brats
from utils import BraTSDataWrapper, brats_collate

def parse_args():
    parser = argparse.ArgumentParser(description='Test and evaluate BraTS models with adversarial attacks')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, 
                       default='/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training',
                       help='Path to BraTS data root directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    parser.add_argument('--test_size', type=float, default=0.3, 
                       help='Fraction of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.5, 
                       help='Fraction of remaining data to use for validation')
    parser.add_argument('--random_state', type=int, default=42, 
                       help='Random seed for data splitting')
    
    # Model paths
    parser.add_argument('--detector_I_path', type=str, 
                       default='/kaggle/input/required3/BraTS_DAE2D_I_final.pth',
                       help='Path to DAE Model I weights')
    parser.add_argument('--detector_II_path', type=str, 
                       default='/kaggle/input/required3/BraTS_DAE2D_II_final.pth',
                       help='Path to DAE Model II weights')
    parser.add_argument('--classifier_path', type=str, 
                       default='/kaggle/input/required3/BraTs_UNet_actual_params.pth',
                       help='Path to UNet classifier weights')
    
    # Model parameters
    parser.add_argument('--input_shape', type=str, default='4,240,240', 
                       help='Input shape as comma-separated values (C,H,W)')
    parser.add_argument('--v_noise', type=float, default=0.05, 
                       help='Noise level for denoising autoencoder')
    parser.add_argument('--activation', type=str, default='leaky_relu', 
                       choices=['relu', 'leaky_relu', 'tanh'], 
                       help='Activation function')
    
    # Structure parameters
    parser.add_argument('--structure_I', type=str, 
                       default='16,max,32,max,linear_bottleneck,256',
                       help='Structure for Model I as comma-separated values (numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck)')
    parser.add_argument('--structure_II', type=str, 
                       default='16,max,32,max,64,max,128,max,linear_bottleneck,128',
                       help='Structure for Model II as comma-separated values (numbers for channels, "max" for maxpool, "linear_bottleneck" for bottleneck)')
    parser.add_argument('--reformer_structure', type=str, 
                       default='16,max,32,max,64,max,128,max,linear_bottleneck,128',
                       help='Structure for reformer model as comma-separated values')
    
    # Attack parameters
    parser.add_argument('--attack_type', type=str, default='fgsm', 
                       choices=['fgsm', 'pgd', 'cw'], 
                       help='Type of adversarial attack')
    parser.add_argument('--epsilons', type=str, default='0.005,0.025,0.05,0.075,0.1', 
                       help='Epsilon values for attacks as comma-separated list')
    parser.add_argument('--num_attack_samples', type=int, default=50, 
                       help='Number of samples to generate for each attack')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/brats_attack_data/', 
                       help='Directory to save attack data')
    parser.add_argument('--load_dir', type=str, default='/kaggle/working/brats_attack_data/', 
                       help='Directory to load attack data from')
    parser.add_argument('--drop_rate', type=str, default='0.1,0.1', 
                       help='Drop rates for detectors as comma-separated list (I,II)')
    parser.add_argument('--graph_name', type=str, default='brats_fgsm_epsilon_analysis', 
                       help='Name for the epsilon analysis graph')
    
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Parse input shape and other parameters
    input_shape = tuple(map(int, args.input_shape.split(',')))
    epsilons = [float(x.strip()) for x in args.epsilons.split(',')]
    drop_rates = [float(x.strip()) for x in args.drop_rate.split(',')]
    drop_rate_dict = {"I": drop_rates[0], "II": drop_rates[1]}
    
    # Parse structures
    structure_I = parse_structure(args.structure_I)
    structure_II = parse_structure(args.structure_II)
    reformer_structure = parse_structure(args.reformer_structure)
    
    print(f"Input shape: {input_shape}")
    print(f"Epsilons: {epsilons}")
    print(f"Drop rates: {drop_rate_dict}")
    print(f"Model I structure: {structure_I}")
    print(f"Model II structure: {structure_II}")
    print(f"Reformer structure: {reformer_structure}")
    
    # ---- Load BraTS-optimized models ----
    detector_I = AEDetector2D(
        DAE,
        args.detector_I_path,
        p=1,
        model_kwargs={
            'image_shape': input_shape,
            'structure': structure_I,
            'v_noise': args.v_noise,
            'activation': args.activation
        }
    )

    detector_II = AEDetector2D(
        DAE,
        args.detector_II_path,
        p=1,
        model_kwargs={
            'image_shape': input_shape,
            'structure': structure_II,
            'v_noise': args.v_noise,
            'activation': args.activation
        }
    )

    classifier = ClassifierUNet(
        UNetPP,
        args.classifier_path,
        model_kwargs={
            'in_channels': 4,
            'out_channels': 4,  # For tumor sub-regions
            'features': [64, 128, 256]
        }
    )
    
    reformer = SimpleReformer2D(
        DAE,
        args.detector_II_path,
        device = classifier.device,
        model_kwargs={
            'image_shape': input_shape,
            'structure': reformer_structure
        }
    )

    # ---- Initialize BraTS pipeline components ----
    detector_dict = {
        "I": detector_I,
        "II": detector_II
    }

    patients = collect_patient_info_from_root(args.data_root, grade_subfolders=True)

    train_patients, temp = train_test_split(patients, test_size=args.test_size, random_state=args.random_state)

    val_patients, test_patients = train_test_split(temp, test_size=args.val_size, random_state=args.random_state)

    train_dataset = BraTSDataset(train_patients, transform=get_train_transforms())

    val_dataset = BraTSDataset(val_patients, transform=get_val_transforms())

    test_dataset = BraTSDataset(test_patients, transform=get_val_transforms())

    # Create DataLoaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=brats_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    os.makedirs(args.save_dir, exist_ok=True)

    # Generate 100 random BraTS volume indices
    idx = torch.randperm(len(test_dataset))[:100]
    batch = next(iter(test_loader))
    X_clean, targets = batch[0], batch[1]
    X_clean = normalize_brats(X_clean)  # Medical image normalization

    # Generate volumetric adversarial examples
    for eps in epsilons:
        attack_data = generate_attack_data_brats(
            classifier.model, 
            args.attack_type, 
            eps,
            num_samples=args.num_attack_samples,
            dataset=test_dataset
        )
        save_obj(attack_data.data.cpu(), f"{args.attack_type}_{eps}_attack", directory=args.save_dir)
        save_obj(attack_data.labels.cpu(), f"{args.attack_type}_{eps}_labels", directory=args.save_dir)

    # ---- Medical imaging evaluation setup ----
    device = next(classifier.model.parameters()).device

    def load_and_normalize_attack(attack_name):
        images = load_obj(f"{attack_name}_attack", args.load_dir).to(device)
        labels = load_obj(f"{attack_name}_labels", args.load_dir).to(device)
        return AttackDataBraTS(normalize_brats(images), labels, name=attack_name)

    # Initialize evaluator with tumor segmentation metrics
    initial_attack = load_and_normalize_attack(f"{args.attack_type}_{epsilons[0]}")
    evaluator = EvaluatorBraTS(operator, initial_attack)

    # Generate clinical performance visualization
    evaluator.plot_epsilon_sweep(
        epsilons=epsilons,
        drop_rate=drop_rate_dict,
        graph_name=args.graph_name,
        attack_type=args.attack_type
    )

if __name__ == "__main__":
    main()

