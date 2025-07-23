import argparse
import torch
import torch.nn as nn
import sys, os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Go up 3 levels to reach the root (where set_root_path.py is)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from set_root_path import add_project_root
add_project_root()

from MEDMNIST.dataloader import load_medmnist
from MEDMNIST.model.Reformer.Classifier.UNet import UNet
from MEDMNIST.model.Reformer.GAN_Recon.GAN import Generator as GANReconGenerator
from MEDMNIST.model.Reformer.GAN_Topo.GAN import Generator as GANTopoGenerator
from MEDMNIST.model.Reformer.GAN_Topo_Recon.GAN import Generator as GANTopoReconGenerator
from MEDMNIST.Attack_generation import generate_mixed_dataset

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.squeeze().to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
    except Exception:
        auc = float('nan')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, auc, f1, cm

def evaluate_with_reformer(reformer, classifier, dataloader, device, num_classes):
    reformer.eval()
    classifier.eval()
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.squeeze().to(device)
            reformed = reformer(images)
            outputs = classifier(reformed)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
    except Exception:
        auc = float('nan')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    return acc, auc, f1, cm

def print_metrics(name, acc, auc, f1, cm):
    print(f"\n[{name}] Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro AUC: {auc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

def main():
    parser = argparse.ArgumentParser(description='Test UNet classifier and reformer+classifier on MedMNIST')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--data_flag', type=str, default='pathmnist', help='MedMNIST dataset flag (e.g., pathmnist, bloodmnist, etc.)')
    parser.add_argument('--test_reformer', type=str, default='all', choices=['all', 'gan_recon', 'gan_topo', 'gan_topo_recon', 'none'], help='Which reformer(s) to test')
    parser.add_argument('--classifier_weights', type=str, default='MEDMNIST/model/Reformer/Classifier/unet_classifier.pth', help='Path to classifier weights file')
    parser.add_argument('--gan_recon_weights', type=str, default='MEDMNIST/model/Reformer/GAN_Recon/gan_recon.pth', help='Path to GAN_Recon reformer weights file')
    parser.add_argument('--gan_topo_weights', type=str, default='MEDMNIST/model/Reformer/GAN_Topo/gan_topo.pth', help='Path to GAN_Topo reformer weights file')
    parser.add_argument('--gan_topo_recon_weights', type=str, default='MEDMNIST/model/Reformer/GAN_Topo_Recon/gan_topo_recon.pth', help='Path to GAN_Topo_Recon reformer weights file')
    parser.add_argument('--pure_adv', action='store_true', help='Use only adversarial examples in test set')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_loader, _, test_loader, info = load_medmnist(data_flag=args.data_flag, batch_size=args.batch_size)
    
    if args.pure_adv:
        print("[INFO] Generating adversarial test set...")
        classifier = UNet(input_channels=info.get('n_channels', 3), num_classes=len(info['label'])).to(device)
        classifier.load_state_dict(torch.load(args.classifier_weights, map_location=device))
        test_loader = generate_mixed_dataset(model=classifier, train_loader=test_loader, device=device, pure_adv=True)
    input_channels = info.get('n_channels', 3)
    num_classes = len(info['label'])

    # Load classifier
    classifier = UNet(input_channels=input_channels, num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load(args.classifier_weights, map_location=device))
    classifier.eval()

    # Evaluate pure classifier
    acc, auc, f1, cm = evaluate(classifier, test_loader, device, num_classes)
    print_metrics("Pure Classifier", acc, auc, f1, cm)

    # Evaluate with reformers
    reformer_types = []
    if args.test_reformer == 'all':
        reformer_types = ['gan_recon', 'gan_topo', 'gan_topo_recon']
    elif args.test_reformer != 'none':
        reformer_types = [args.test_reformer]

    for reformer_type in reformer_types:
        if reformer_type == 'gan_recon':
            Generator = GANReconGenerator
            weights = args.gan_recon_weights
        elif reformer_type == 'gan_topo':
            Generator = GANTopoGenerator
            weights = args.gan_topo_weights
        elif reformer_type == 'gan_topo_recon':
            Generator = GANTopoReconGenerator
            weights = args.gan_topo_recon_weights
        else:
            continue
        # Load reformer
        reformer = Generator(output_channels=input_channels).to(device)
        checkpoint = torch.load(weights, map_location=device)
        if isinstance(checkpoint, dict) and "generator" in checkpoint:
            reformer.load_state_dict(checkpoint["generator"])
        else:
            reformer.load_state_dict(checkpoint)

        reformer.eval()
        acc, auc, f1, cm = evaluate_with_reformer(reformer, classifier, test_loader, device, num_classes)
        print_metrics(f"Reformer+Classifier: {reformer_type}", acc, auc, f1, cm)

if __name__ == '__main__':
    main()
# python test.py --batch_size 32 --device cuda --data_flag pathmnist --test_reformer all
#python test.py --data_flag pathmnist --test_reformer none --pure_adv

