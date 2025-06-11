import sys
sys.path.append('/kaggle/input/required')
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from brats_loader import get_brats_dataloaders, dice_coefficient, hausdorff_distance_95
import torch.nn.functional as F 
  
# --- Training and Evaluation Functions ---
def evaluate(model, dataloader, device):
    """Evaluates the segmentation model on a given dataloader."""
    model.eval()
    val_losses = []
    val_dice_scores = {'WT': [], 'TC': [], 'ET': []}
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            images = batch_data['image'].to(device)
            masks_true = batch_data['mask'].to(device)  # (Batch, 3, H, W)

            outputs = model(images)  # (Batch, 3, H, W) logits
            
            # Convert one-hot to class indices
            true_masks = torch.argmax(masks_true, dim=1)  # (Batch, H, W)
            
            loss = criterion(outputs, true_masks)
            val_losses.append(loss.item())

            probabilities = torch.softmax(outputs, dim=1)
            predicted_masks = torch.argmax(probabilities, dim=1)  # (Batch, H, W)

            masks_true_np = true_masks.cpu().numpy()
            predicted_masks_np = predicted_masks.cpu().numpy()

            for i in range(masks_true_np.shape[0]):
                true_mask = masks_true_np[i]
                pred_mask = predicted_masks_np[i]

                # Calculate Dice for each tumor region
                wt_true = (true_mask == 1) | (true_mask == 2) | (true_mask == 3)
                wt_pred = (pred_mask == 1) | (pred_mask == 2) | (pred_mask == 3)
                val_dice_scores['WT'].append(dice_coefficient(wt_true, wt_pred))

                tc_true = (true_mask == 1) | (true_mask == 3)
                tc_pred = (pred_mask == 1) | (pred_mask == 3)
                val_dice_scores['TC'].append(dice_coefficient(tc_true, tc_pred))

                et_true = (true_mask == 3)
                et_pred = (pred_mask == 3)
                val_dice_scores['ET'].append(dice_coefficient(et_true, et_pred))

    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_dice_results = {
        'WT': np.mean(val_dice_scores['WT']) if val_dice_scores['WT'] else 0.0,
        'TC': np.mean(val_dice_scores['TC']) if val_dice_scores['TC'] else 0.0,
        'ET': np.mean(val_dice_scores['ET']) if val_dice_scores['ET'] else 0.0,
    }
    return avg_val_loss, avg_dice_results

def train_model(train_loader, val_loader, file_name, num_epochs=50, device=None):
    """Trains the 2D UNet++ segmentation model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize UNet++ with BraTS-specific parameters
    model = UNetPP(
        in_channels=4,        # T1, T1ce, T2, FLAIR
        out_channels=3,       # NCR/NET, ED, ET
        features=[64, 128, 256]  # Reduced depth for BraTS
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added regularization

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data['image'].to(device)
            masks_true = batch_data['mask'].to(device)

            # Convert one-hot to class indices
            true_masks = torch.argmax(masks_true, dim=1)  # (Batch, H, W)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, true_masks)
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

    if file_name is not None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    return model

if __name__ == '__main__':
    # --- Configuration ---
    BRATS_TRAIN_DATA_ROOT = '/kaggle/input/miccaibrats2019/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training'
    MODEL_OUTPUT_FILE = "models/brats_segmentation_model_local_split_unetpp.pth"
    NUM_EPOCHS = 5
    BATCH_SIZE = 4  # Reduced for UNet++ memory requirements
    RANDOM_STATE = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Get DataLoaders ---
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
        num_epochs=NUM_EPOCHS,
        device=device
    )

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on Local Test Set ---")
    loaded_model = UNetPP(in_channels=4, out_channels=3, features=[64, 128, 256]).to(device)
    loaded_model.load_state_dict(torch.load(MODEL_OUTPUT_FILE))
    
    test_loss, test_dice_results = evaluate(loaded_model, local_test_loader, device)
    print(f"Local Test Results: Loss: {test_loss:.4f}, "
          f"Dice WT: {test_dice_results['WT']:.4f}, "
          f"Dice TC: {test_dice_results['TC']:.4f}, "
          f"Dice ET: {test_dice_results['ET']:.4f}")
