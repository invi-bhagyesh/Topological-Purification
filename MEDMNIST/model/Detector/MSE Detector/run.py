import argparse
import torch
from Simple_Autoencoder import SimpleAutoencoder
from Perturbation_classifier import PerturbationClassifier
from Simple_Autoencoder_trainer import train_autoencoder
from Perturbation_classifier_trainer import train_classifier
from MEDMNIST.dataloader import load_medmnist
from MEDMNIST.Attack_generation import SimpleCNN, train_model, generate_mixed_dataset


def get_autoencoder_data_loaders(batch_size):
    train_loader, val_loader, _, _ = load_medmnist(batch_size=batch_size)
    return train_loader, val_loader

def get_classifier_data_loader(classifier_model, batch_size, device):
    train_loader, _, _, _ = load_medmnist(batch_size=batch_size)
    mixed_loader = generate_mixed_dataset(classifier_model, train_loader, device=device)
    return mixed_loader

def get_or_train_attack_classifier(batch_size, device, epochs=5, lr=0.001):
    model = SimpleCNN(num_classes=9)
    try:
        model.load_state_dict(torch.load('attack_classifier.pth', map_location=device))
        print("Loaded pre-trained attack classifier.")
    except Exception:
        print("Training attack classifier from scratch...")
        train_loader, _, _, _ = load_medmnist(batch_size=batch_size)
        train_model(model, train_loader, device=device, epochs=epochs, lr=lr)
        torch.save(model.state_dict(), 'attack_classifier.pth')
        print("Saved attack classifier to attack_classifier.pth")
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='MSE Detector Training Pipeline')
    parser.add_argument('--train_autoencoder', action='store_true', help='Train the autoencoder')
    parser.add_argument('--train_classifier', action='store_true', help='Train the classifier on reconstructed images')
    parser.add_argument('--epochs_autoencoder', type=int, default=30, help='Epochs for autoencoder training')
    parser.add_argument('--epochs_classifier', type=int, default=20, help='Epochs for classifier training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr_autoencoder', type=float, default=1e-3, help='Learning rate for autoencoder')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='Learning rate for classifier')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='Loss type for autoencoder')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    autoencoder = None
    if args.train_autoencoder:
        train_loader, val_loader = get_autoencoder_data_loaders(args.batch_size)
        autoencoder = train_autoencoder(
            train_loader, val_loader, device,
            epochs=args.epochs_autoencoder, lr=args.lr_autoencoder, loss_type=args.loss_type
        )
    else:
        autoencoder = SimpleAutoencoder().to(device)
        autoencoder.load_state_dict(torch.load('mse_autoencoder.pth', map_location=device))
        autoencoder.eval()

    if args.train_classifier:
        attack_classifier = get_or_train_attack_classifier(args.batch_size, device)
        mixed_loader = get_classifier_data_loader(attack_classifier, args.batch_size, device)
        train_classifier(
            autoencoder, mixed_loader, device,
            epochs=args.epochs_classifier, lr=args.lr_classifier
        )

if __name__ == '__main__':
    main() 