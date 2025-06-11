# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import wandb # Assuming you use Weights & Biases for experiment tracking
import os
import numpy as np
# from brats_loader import get_brats_dataloaders, dice_coefficient, hausdorff_distance_95
import torch.nn.functional as F 
# from model.clf_3d_unet import CNNClassifier3D # Example import
import time


# --- Training and Evaluation Functions ---
def evaluate(model, dataloader, device):
    """
    Evaluates the segmentation model on a given dataloader.
    (This function assumes the model outputs logits for 3 channels and
     calculates Dice for WT, TC, ET as defined previously.)
    """
    model.eval()
    val_losses = []
    val_dice_scores = {'WT': [], 'TC': [], 'ET': []}
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            images = batch_data['image'].to(device)
            masks_true = batch_data['mask'].to(device) # (Batch, 3, H, W)

            outputs = model(images) # (Batch, 3, H, W) logits

            loss = criterion(outputs, masks_true)
            val_losses.append(loss.item())

            probabilities = torch.sigmoid(outputs)
            predicted_masks_binary = (probabilities > 0.5).float()

            masks_true_np = masks_true.cpu().numpy()
            predicted_masks_np = predicted_masks_binary.cpu().numpy()

            for i in range(masks_true_np.shape[0]):
                true_mask_slice = masks_true_np[i]
                pred_mask_slice = predicted_masks_np[i]

                wt_true = (true_mask_slice[0] + true_mask_slice[1] + true_mask_slice[2]) > 0.5
                wt_pred = (pred_mask_slice[0] + pred_mask_slice[1] + pred_mask_slice[2]) > 0.5
                val_dice_scores['WT'].append(dice_coefficient(wt_true, wt_pred))

                tc_true = (true_mask_slice[0] + true_mask_slice[2]) > 0.5
                tc_pred = (pred_mask_slice[0] + pred_mask_slice[2]) > 0.5
                val_dice_scores['TC'].append(dice_coefficient(tc_true, tc_pred))

                et_true = true_mask_slice[2] > 0.5
                et_pred = pred_mask_slice[2] > 0.5
                val_dice_scores['ET'].append(dice_coefficient(et_true, et_pred))

    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_dice_results = {
        'WT': np.mean(val_dice_scores['WT']) if val_dice_scores['WT'] else 0.0,
        'TC': np.mean(val_dice_scores['TC']) if val_dice_scores['TC'] else 0.0,
        'ET': np.mean(val_dice_scores['ET']) if val_dice_scores['ET'] else 0.0,
    }
    return avg_val_loss, avg_dice_results


def train_model(train_loader, val_loader, file_name, model_params, num_epochs=50, device=None):
    """
    Trains a segmentation model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D(**model_params).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)

    # wandb.init(project="brats_segmentation_separate_files", config={
    #     "learning_rate": 0.01,
    #     "epochs": num_epochs,
    #     "batch_size": train_loader.batch_size,
    #     "model_params": model_params,
    #     "optimizer": "SGD",
    #     "dataset": "BraTS_LocalSplit"
    # })

    best_val_loss = float('inf')
    patience = 3  # Number of epochs to wait before stopping
    epochs_without_improvement = 0
    best_model_state = None

    
    print("Starting training...")
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            

            images = batch_data['image'].to(device)
            masks_true = batch_data['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks_true)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        val_loss, val_dice_results = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice WT: {val_dice_results['WT']:.4f}, "
              f"Val Dice TC: {val_dice_results['TC']:.4f}, "
              f"Val Dice ET: {val_dice_results['ET']:.4f}")

        # wandb.log({
        #     "epoch": epoch + 1,
        #     "loss/train": train_loss,
        #     "loss/val": val_loss,
        #     "dice/val_WT": val_dice_results['WT'],
        #     "dice/val_TC": val_dice_results['TC'],
        #     "dice/val_ET": val_dice_results['ET'],
        # })

                # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

      

  
    # --- Restore and Save the Best Model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best model based on validation loss.")
    
    if file_name is not None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(model.state_dict(), file_name)
        print(f"Best model saved to {file_name}")



    
    # wandb.finish()
    return model


if __name__ == '__main__':
    # --- Configuration ---
    BRATS_TRAIN_DATA_ROOT = '/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training'
    MODEL_OUTPUT_FILE = "./brats_segmentation_model_local_split.pth"
    MODEL_PARAMS = {
    "in_channels": 4,
    "features": [32, 64, 128, 256],
    "out_channels": 3
} # Example: 4 input channels, 3 output channels
    NUM_EPOCHS = 6
    BATCH_SIZE = 8
    RANDOM_STATE = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Get DataLoaders from data_setup.py ---
    train_loader, val_loader, local_test_loader = get_brats_dataloaders(
        train_data_root=BRATS_TRAIN_DATA_ROOT,
        train_ratio=0.70,
        val_ratio=0.15,
        local_test_ratio=0.15,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count()
    )

    # --- Start Training ---
    trained_model = train_model(
        train_loader,
        val_loader,
        MODEL_OUTPUT_FILE,
        model_params=MODEL_PARAMS,
        num_epochs=NUM_EPOCHS,
        device=device
    )
    # --- Optional: Final Evaluation on Local Test Set ---
    print("\n--- Final Evaluation on Local Test Set ---")
    loaded_model = UNet3D(**MODEL_PARAMS).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_OUTPUT_FILE))
    
    test_loss, test_dice_results = evaluate(loaded_model, local_test_loader, device)
    print(f"Local Test Results: Loss: {test_loss:.4f}, "
          f"Dice WT: {test_dice_results['WT']:.4f}, "
          f"Dice TC: {test_dice_results['TC']:.4f},"
          f"Dice ET: {test_dice_results['ET']:.4f}")
