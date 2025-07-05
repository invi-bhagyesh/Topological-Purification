import sys
import matplotlib
from scipy.stats import entropy
from numpy.linalg import norm
from matplotlib.ticker import FuncFormatter
from monai.metrics import compute_hausdorff_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pylab
import os
import matplotlib.pyplot as plt




class EvaluatorBraTS:
    def __init__(self, operator, untrusted_data, graph_dir="/kaggle/working/graphs"):
        """
        Evaluator for BraTS medical imaging defense strategies.

        operator: OperatorBraTS object
        untrusted_data: Adversarial/noisy BraTS data (3D volumes or 2D slices)
        graph_dir: Output directory for evaluation metrics
        """
        self.operator = operator
        self.untrusted_data = self.prepare_brats_data(untrusted_data)
        self.graph_dir = graph_dir
        os.makedirs(self.graph_dir, exist_ok=True)
        self.data_package = operator.operate(self.untrusted_data)
       

    def load_data(self, new_data):
        """Update evaluation data and reprocess through defense pipeline"""
        self.untrusted_data = self.prepare_brats_data(new_data)
        self.data_package = self.operator.operate(self.untrusted_data)

    def prepare_brats_data(self, data):
        device = next(self.operator.classifier.model.parameters()).device
        normalized_data = data.data.to(device)
        normalized_data = torch.clamp(normalized_data, 0.0, 1.0)
        
        # Changed from AttackData to AttackDataBraTS
        return AttackDataBraTS(normalized_data, data.labels.to(device), data.name)

    def get_normal_acc(self, normal_all_pass):
        """Volumetric accuracy calculation for BraTS"""
        normal_tups = self.operator.normal
        num_normal = len(normal_tups)
        
        # For 3D data: require correct predictions for all slices
        if self.operator.normal[0][0].ndim == 3:  # 3D volume
            both_acc = sum(np.all(filt_tup, axis=1).mean() for filt_tup in normal_tups[normal_all_pass])
        else:  # 2D slices
            both_acc = sum(1 for _, XpC in normal_tups[normal_all_pass] if XpC) / num_normal

        return both_acc, 0, 0, 0  # Simplified for medical imaging context

    def get_attack_acc(self, attack_pass):
        attack_tups = self.data_package
        
        if len(attack_tups) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        def medical_dice_score(pred_mask, target_mask):
            """BraTS-optimized Dice calculation"""
            dice_scores = []
            
            # Match model's 3 output channels (background, tumor_core, enhancing_tumor)
            for class_id in [1, 2, 3]:  # Skip background (0)
                pred_class = (pred_mask == class_id).astype(np.float32)
                target_class = (target_mask == class_id).astype(np.float32)
                
                intersection = (pred_class * target_class).sum()
                union = pred_class.sum() + target_class.sum()
                
                if union > 0:
                    dice = (2.0 * intersection) / (union + 1e-8)
                    dice_scores.append(dice)
                else:
                    # Handle empty tumor regions in prediction/target
                    dice_scores.append(1.0 if (pred_class.sum() + target_class.sum()) == 0 else 0.0)
            
            return np.mean(dice_scores)
        
        original_dice = []
        healed_dice = []
        original_hd95 = []
        healed_hd95 = []
        
        for orig_pred, healed_pred, target in attack_tups:
            # Convert to binary tumor masks
            orig_pred_bin = (orig_pred > 0).astype(np.float32)
            healed_pred_bin = (healed_pred > 0).astype(np.float32)
            target_bin = (target > 0).astype(np.float32)
            
            def safe_hd95(pred, target):
                """Medical-safe HD95 calculation"""
                # Case 1: No tumor in ground truth
                if np.sum(target) == 0:
                    return 0.0  # Perfect score if no tumor exists
                
                # Case 2: Model failed to detect tumor
                if np.sum(pred) == 0:
                    return 300.0  # Penalty value (300mm = 30cm brain size)
                
                # Case 3: Valid tumor regions
                device = next(self.operator.classifier.model.parameters()).device
                pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).to(device)
                target_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(device)
                return compute_hausdorff_distance(
                    pred_tensor, target_tensor, 
                    percentile=95,
                    include_background=False
                ).item()
            
            hd95_orig = safe_hd95(orig_pred_bin, target_bin)
            hd95_healed = safe_hd95(healed_pred_bin, target_bin)
                
            # Compute Dice
            orig_dice = medical_dice_score(orig_pred, target)
            healed_dice_score = medical_dice_score(healed_pred, target)
            
            original_dice.append(orig_dice)
            healed_dice.append(healed_dice_score)
            original_hd95.append(hd95_orig)
            healed_hd95.append(hd95_healed)
        
        return (
            np.mean(original_dice), 
            np.mean(healed_dice),
            np.mean(original_hd95),
            np.mean(healed_hd95)
        )
        
    def plot_epsilon_sweep(self, graph_name, drop_rate, epsilons, 
                          attack_type="fgsm", data_dir='/kaggle/working/brats_attack_data/'):
        
        metrics = {
            'original_dice': [],
            'healed_dice': [],
            'original_hd95': [],
            'healed_hd95': [],
            'detection_rate': []
        }
        
        plt.figure(figsize=(12, 8))
        device = next(self.operator.classifier.model.parameters()).device
        
        # Calculate thresholds once
        thresholds = self.operator.get_thrs(drop_rate)
        print(f"Using thresholds: {thresholds}")
        
        for eps in epsilons:
            print(f"\n=== DEBUGGING ε={eps} ===")
            
            # Load attack data
            attack_name = f"{attack_type}_{eps}"
            attack_data = load_obj(f"{attack_name}_attack", data_dir).to(device)
            attack_labels = load_obj(f"{attack_name}_labels", data_dir).to(device)
            
            print(f"Loaded {len(attack_data)} samples")
            
            # Process through defense pipeline
            attack_dataset = AttackDataBraTS(attack_data, attack_labels, name=attack_name)
            self.load_data(attack_dataset)
            
            # Test detector marks
            detector_marks = {}
            for name, detector in self.operator.det_dict.items():
                marks = detector.mark(attack_data[:5])  # Test first 5 samples
                detector_marks[name] = marks
                print(f"Detector {name} sample marks: {marks}")
            
            # Get filtering results
            attack_pass, stats = self.operator.filter(attack_dataset.data, thresholds)
            print(f"Filter stats: {stats}")
            print(f"Samples passed: {len(attack_pass)}/{len(attack_data)}")
            
            # Calculate metrics
            orig_dice, healed_dice, orig_hd95, healed_hd95 = self.get_attack_acc(attack_pass)
            detection_rate = 1.0 - (len(attack_pass) / len(attack_data))
            
            print(f"Original Dice: {orig_dice:.4f}")
            print(f"Healed Dice: {healed_dice:.4f}") 
            print(f"Original HD95: {orig_hd95:.2f} mm")
            print(f"Healed HD95: {healed_hd95:.2f} mm")
            print(f"Detection Rate: {detection_rate:.4f}")
            
            # Store results
            metrics['original_dice'].append(orig_dice)
            metrics['healed_dice'].append(healed_dice)
            metrics['original_hd95'].append(orig_hd95)
            metrics['healed_hd95'].append(healed_hd95)
            metrics['detection_rate'].append(detection_rate)

        valid_hd = [x for x in metrics['original_hd95'] + metrics['healed_hd95'] if np.isfinite(x)]
        max_hd = max(valid_hd) if valid_hd else 50.0  # 50mm clinical safety threshold
        
        
        plt.plot(epsilons, metrics['original_dice'], 'r--', marker='o', linewidth=2, label="Original Dice")
        plt.plot(epsilons, metrics['healed_dice'], 'g-', marker='s', linewidth=2, label="Healed Dice")
        plt.plot(epsilons, metrics['detection_rate'], 'b:', marker='^', linewidth=2, label="Detection Rate")
                
        plt.xlabel("Attack Strength (ε)", fontsize=12)
        plt.ylabel("Dice Coefficient", fontsize=12)
        plt.title("BraTS Defense Performance vs Adversarial Strength", fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)
                
        plt.savefig(f"{self.graph_dir}/{graph_name}_brats.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(epsilons, metrics['original_hd95'], 'm--', marker='x', linewidth=2, label="Original HD95")
        plt.plot(epsilons, metrics['healed_hd95'], 'c-', marker='+', linewidth=2, label="Healed HD95")
        
        plt.xlabel("Attack Strength (ε)", fontsize=12)
        plt.ylabel("Hausdorff Distance (mm)", fontsize=12)
        plt.title("Tumor Boundary Accuracy vs Adversarial Strength", fontsize=14)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max_hd * 1.1)
        plt.yticks(np.arange(0, max_hd*1.1, 20))
        
        plt.savefig(f"{self.graph_dir}/{graph_name}_hausdorff.png", dpi=300, bbox_inches='tight')
        plt.close()