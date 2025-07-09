# MEDMNIST Detector Pipelines

This directory contains modular pipelines for adversarial detection on the MedMNIST dataset using two approaches:

- **MSE Detector** (reconstruction-based)
- **Contrastive Detector** (representation-based)

Each approach is organized in its own folder with a unified `run.py` script and modular trainer/model files.

---

## Folder Structure

```
MEDMNIST/
  model/
    Detector/
      MSE Detector/
        run.py
        Simple_Autoencoder.py
        Simple_Autoencoder_trainer.py
        Perturbation_classifier.py
        Perturbation_classifier_trainer.py
      Contrastive Detector/
        run.py
        ContrastiveEncoder.py
        Contrastive_trainer.py
        Perturbation_classifier.py
```

---



## Usage

### **MSE Detector**
- **Train autoencoder and classifier:**
  ```bash
  cd model/Detector/MSE\ Detector
  python run.py --train_autoencoder --train_classifier
  ```
- **Train only autoencoder:**
  ```bash
  python run.py --train_autoencoder
  ```
- **Train only classifier (requires pre-trained autoencoder):**
  ```bash
  python run.py --train_classifier
  ```
- **Options:**
  - `--epochs_autoencoder`, `--epochs_classifier`, `--batch_size`, `--lr_autoencoder`, `--lr_classifier`, `--loss_type mse|l1`, `--device cuda|cpu`

### **Contrastive Detector**
- **Train encoder and classifier:**
  ```bash
  cd model/Detector/Contrastive\ Detector
  python run.py --train_encoder --train_classifier
  ```
- **Train only encoder:**
  ```bash
  python run.py --train_encoder
  ```
- **Train only classifier (requires pre-trained encoder):**
  ```bash
  python run.py --train_classifier
  ```
- **Options:**
  - `--epochs_encoder`, `--epochs_classifier`, `--batch_size`, `--lr_encoder`, `--lr_classifier`, `--device cuda|cpu`

---

## Modular Structure

- **Model files** define the neural network architectures.
- **Trainer files** contain reusable training functions (can be run standalone for debugging).
- **run.py** orchestrates the full pipeline and exposes command-line arguments for flexible experimentation.

---

## Notes
- Adversarial examples for classifier training are generated using a pre-trained attack classifier (trained or loaded automatically).
- You can further customize data augmentations, loss functions, and evaluation as needed.

---
