import argparse
import torch
from ContrastiveEncoder import ContrastiveEncoder
from Perturbation_classifier import ContrastiveClassifier
from Contrastive_trainer import train_contrastive_encoder, train_contrastive_classifier
from MEDMNIST.dataloader import load_medmnist
from MEDMNIST.Attack_generation import SimpleCNN, train_model, generate_mixed_dataset


def get_encoder_data_loaders(batch_size):
    train_loader, val_loader, _, _ = load_medmnist(batch_size=batch_size)
    return train_loader, val_loader

def get_classifier_data_loader(classifier_model, batch_size, device):
    # Use the trained classifier model to generate adversarial examples
    train_loader, _, _, _ = load_medmnist(batch_size=batch_size)
    mixed_loader = generate_mixed_dataset(classifier_model, train_loader, device=device)
    return mixed_loader

def get_or_train_attack_classifier(batch_size, device, epochs=5, lr=0.001):
    # This model is used to generate adversarial examples
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
    parser = argparse.ArgumentParser(description='Contrastive Model Training Pipeline')
    parser.add_argument('--train_encoder', action='store_true', help='Train the contrastive encoder')
    parser.add_argument('--train_classifier', action='store_true', help='Train the classifier on embeddings')
    parser.add_argument('--epochs_encoder', type=int, default=30, help='Epochs for encoder training')
    parser.add_argument('--epochs_classifier', type=int, default=20, help='Epochs for classifier training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='Learning rate for encoder')
    parser.add_argument('--lr_classifier', type=float, default=1e-3, help='Learning rate for classifier')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    encoder = None
    if args.train_encoder:
        train_loader, val_loader = get_encoder_data_loaders(args.batch_size)
        encoder = train_contrastive_encoder(
            train_loader, val_loader, device,
            epochs=args.epochs_encoder, lr=args.lr_encoder
        )
    else:
        encoder = ContrastiveEncoder().to(device)
        encoder.load_state_dict(torch.load('contrastive_encoder.pth', map_location=device))
        encoder.eval()

    if args.train_classifier:
        # Get or train the attack classifier for adversarial example generation
        attack_classifier = get_or_train_attack_classifier(args.batch_size, device)
        mixed_loader = get_classifier_data_loader(attack_classifier, args.batch_size, device)
        train_contrastive_classifier(
            encoder, mixed_loader, device,
            epochs=args.epochs_classifier, lr=args.lr_classifier
        )

if __name__ == '__main__':
    main() 