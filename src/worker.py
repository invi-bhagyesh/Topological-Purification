import sys
import matplotlib
matplotlib.use('Agg')
from scipy.stats import entropy
from numpy.linalg import norm
from matplotlib.ticker import FuncFormatter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pylab
import os
import utils as utils
import matplotlib.pyplot as plt

# --- CONSTANTS for BraTS Data ---
BRATS_CHANNELS = 4 # T1, T1ce, T2, FLAIR
BRATS_IMG_SIZE = 240 # BraTS images are typically 240x240

class AEDetector:
    def __init__(self, model_class, model_path, p=1, device=None, model_kwargs=None):
        """
        Error-based detector.

        model_class: The class of the AE model (e.g., DenoisingAutoEncoder).
        model_path: Path to the model's saved state_dict (.pth).
        p: Power for error norm (e.g., 1 = L1, 2 = L2).
        device: torch.device object or string ("cuda" / "cpu").
        model_kwargs: Dictionary of keyword arguments to pass to model_class constructor.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.p = p

    def mark(self, X):
        """
        X: A PyTorch tensor of shape [N, C, H, W] on CPU or GPU.
           For BraTS, C=4, H=240, W=240.
        Returns: 1D numpy array of anomaly scores (reconstruction errors).
        """
        self.model.eval()

        with torch.no_grad():
            # No permutation needed if input is already NCHW (N, 4, 240, 240)
            # If your external data source somehow gives NHWC, you'd need:
            # if X.ndim == 4 and X.shape[1:] == torch.Size([BRATS_IMG_SIZE, BRATS_IMG_SIZE, BRATS_CHANNELS]):
            #     X = X.permute(0, 3, 1, 2) # Convert from NHWC to NCHW

            X = X.to(self.device)
            recon = self.model(X)
            diff = torch.abs(X - recon)  # Absolute error
            # Calculate mean error across channels, height, and width for each sample
            score = torch.mean(torch.pow(diff, self.p), dim=[1, 2, 3])
            return score.cpu().numpy()  # Convert to NumPy

    def print(self):
        return "AEDetector:" + os.path.basename(self.model.path) # Assuming your model has a .path attribute


class IdReformer:
    def __init__(self, path="IdentityFunction"):
        """
        Identity reformer. Returns input unchanged.
        """
        self.path = path
        self.heal = lambda X: X  # No transformation

    def print(self):
        return "IdReformer:" + self.path


class SimpleReformer:
    def __init__(self, model_class, model_path, device=None, model_kwargs=None):
        """
        Autoencoder-based reformer. Applies reconstruction (healing).

        model_class: Class definition of the autoencoder.
        model_path: Path to the saved model (.pth).
        device: torch.device or str ("cuda" or "cpu").
        model_kwargs: Dictionary of keyword arguments for model_class constructor.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # Ensure your DAE model has an image_shape attribute if used, or pass it directly.
        # Assuming model_kwargs contains input_shape, or DAE sets it internally.
        # self.image_shape = self.model.image_shape # Uncomment if your DAE exposes this

    def heal(self, X):
        """
        X: Tensor [N, C, H, W] (should be on same device as model)
           For BraTS, C=4, H=240, W=240.
        Returns: Reconstructed input clipped between 0 and 1 (as Torch Tensor)
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)

            # Check if input needs unsqueezing or permuting.
            # BraTS DataLoader output should be NCHW directly.
            # If a single CHW image is passed, add batch dim.
            if X.ndim == 3: # Assuming CHW format for a single image
                X = X.unsqueeze(0) # Add batch dimension to make it NCHW
            # No permutation needed if input is already NCHW (N, 4, 240, 240)
            # if X.ndim == 4 and X.shape[1:] == torch.Size([BRATS_IMG_SIZE, BRATS_IMG_SIZE, BRATS_CHANNELS]):
            #     X = X.permute(0, 3, 1, 2) # Convert from NHWC to NCHW if necessary

            # BraTS images from our data_setup.py are already normalized to [0, 1]
            # so the X / 255.0 might not be needed or should be conditional.
            # Your current DAE training expects [0,1] data, so this check is relevant
            # if the source of X is raw unnormalized data.
            if X.max() > 1.0 + 1e-6: # Add small epsilon to tolerate float precision
                print(f"Warning: Input max value {X.max():.2f} > 1.0. Normalizing by 255.0.")
                X = X / 255.0

            # print(f"Input range to reformer: [{X.min():.6f}, {X.max():.6f}]") # For debugging
            # print(f"Input mean to reformer: {X.mean():.6f}") # For debugging

            recon = self.model(X)
            recon = torch.clamp(recon, 0.0, 1.0) # Clamp output to the expected [0, 1] range

            # print(f"Output range from reformer: [{recon.min():.6f}, {recon.max():.6f}]") # For debugging
            return recon

    def print(self):
        return "SimpleReformer:" + os.path.basename(self.model.path) # Assuming your model has a .path attribute


def JSD(P, Q):
    """
    Jensen-Shannon Divergence between two 1D arrays (P, Q).
    P, Q: Numpy arrays representing discrete distributions (sums to 1).
    """
    # Ensure inputs are treated as probability distributions
    _P = P / (norm(P, ord=1) + 1e-9) # Add epsilon to avoid division by zero if P is all zeros
    _Q = Q / (norm(Q, ord=1) + 1e-9) # Add epsilon
    _M = 0.5 * (_P + _Q)
    # Ensure _M is not zero for entropy calculation
    _M[np.where(_M == 0)] = 1e-9 # Replace zeros with small epsilon
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


class DBDetector:
    def __init__(self, reconstructor, prober, classifier, option="jsd", T=1):
        """
        Divergence-Based Detector using PyTorch.

        reconstructor: One reformer (e.g., SimpleReformer instance).
        prober: Another reformer (same type).
        classifier: Classifier object with classify() method.
        option: Distance measure to use, default is 'jsd'.
        T: Temperature for softmax (scaling logits).
        """
        self.prober = prober
        self.reconstructor = reconstructor
        self.classifier = classifier
        self.option = option
        self.T = T

    def mark(self, X):
        if self.option == "jsd":
            return self.mark_jsd(X)
        else:
            raise NotImplementedError(f"Only 'jsd' is implemented, got {self.option}")

    def mark_jsd(self, X):
        """
        Returns JSD divergence between classifier outputs on
        probed and reconstructed samples.

        X: Input tensor [N, C, H, W] (torch.Tensor)
        """
        # Ensure X is on the correct device for healing
        X = X.to(self.reconstructor.device)

        Xp = self.prober.heal(X)
        Xr = self.reconstructor.heal(X)

        # Classify outputs. For segmentation, the classifier outputs logits (N, C_seg, H, W).
        # We need to turn these into a single "probability distribution" per image.
        # A common way for uncertainty/divergence in segmentation is to flatten the spatial
        # dimensions and then take softmax over the class probabilities for each pixel,
        # or aggregate to image-level class probabilities (e.g., probability of tumor presence).
        #
        # For a DAE setup where the *classifier* is a segmentation model,
        # the 'classify' method needs to produce a 1D probability distribution per image.
        # This is a crucial design choice. For now, I'll assume your classifier's
        # `classify` method already handles reducing the spatial dimensions
        # and producing a (N, num_classes) probability distribution (e.g., tumor vs no_tumor).
        # If your classifier output is still (N, C_seg, H, W), you'll need an aggregation
        # step here before passing to JSD.
        # Example aggregation if output is (N, C_seg, H, W):
        # probs_p = F.softmax(self.classifier.model(Xp) / self.T, dim=1).mean(dim=[-1, -2]) # Average pixel-wise probs
        # Then, pass probs_p.cpu().numpy() to JSD
        
        Pp = self.classifier.classify(Xp, option="prob", T=self.T)  # numpy array [N, num_classes]
        Pr = self.classifier.classify(Xr, option="prob", T=self.T)

        marks = [JSD(Pp[i], Pr[i]) for i in range(len(Pr))]
        return np.array(marks)

    def print(self):
        return "Divergence-Based Detector"


class Classifier:
    def __init__(self, model_class, classifier_path, device=None, model_kwargs=None):
        """
        PyTorch classifier wrapper. Assumes the model outputs raw logits.

        model_class: A callable (e.g., a class or lambda) that returns the classifier architecture.
        classifier_path: Path to saved model weights (.pth).
        device: torch.device or string ("cuda"/"cpu"). Auto-detected if not provided.
        model_kwargs: dict of kwargs for model_class constructor.
        """
        self.path = classifier_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.model.eval()
        # Assuming model_kwargs will contain details about output channels (e.g., 3 for BraTS regions)
        # and that your classifier model outputs (N, C_seg, H, W) for segmentation.

    def classify(self, X, option="logit", T=1):
        """
        Classify input. For BraTS segmentation, X is (N, 4, H, W).
        The model's output `logits` will be (N, C_seg, H, W) where C_seg is your number of segmentation classes (e.g., 3).

        X: Torch tensor [N, C, H, W]
        option: "logit" to return raw logits, "prob" to return softmax probs.
                For segmentation, this implies pixel-wise logits/probs.
        T: Temperature (used only with option="prob")
        Returns: NumPy array. For segmentation, this needs careful handling:
                 Are you returning (N, C_seg, H, W) logits/probs? Or an image-level classification?
                 The `DBDetector` expects [N, num_classes] for JSD.
        """
        # No permutation needed if input is already NCHW (N, 4, 240, 240)
        # if X.ndim == 4 and X.shape[1:] == torch.Size([28, 28, 1]): # MNIST specific, remove or adapt
        #     X = X.permute(0, 3, 1, 2)

        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()

            X = X.to(self.device)
            logits = self.model(X) # This should be (N, C_seg, H, W) for segmentation

            if option == "logit":
                # Returns (N, C_seg, H, W)
                return logits.cpu().numpy()

            elif option == "prob":
                # For segmentation models, F.softmax(logits, dim=1) gives pixel-wise probabilities over classes.
                # To get a *single probability distribution per image* as required by JSD,
                # you need an aggregation strategy.
                # Common strategies:
                # 1. Average class probabilities across spatial dimensions:
                probs = F.softmax(logits / T, dim=1).mean(dim=[-1, -2]) # Average over H and W
                # 2. Probability of whole tumor presence (if applicable to your segmentation classes):
                #    e.g., if class 0 is background, calculate 1 - prob(background)
                #    probs = F.softmax(logits / T, dim=1)
                #    tumor_presence_prob = 1 - probs[:, 0, :, :].mean(dim=[-1, -2]) # Avg background prob, then 1-it
                #    probs = torch.stack([tumor_presence_prob, 1 - tumor_presence_prob], dim=1)
                #
                # For now, I'll use strategy 1 (average pixel-wise class probabilities)
                # This will produce a (N, C_seg) tensor, suitable for JSD.
                return probs.cpu().numpy()

            else:
                raise ValueError(f"Invalid option: {option}. Use 'logit' or 'prob'.")

    def print(self):
        return "Classifier:" + os.path.basename(self.path)


# Key fixes for your MAGNET-inspired defense system

# 1. Add missing utility functions (utils.py)
import pickle
import numpy as np
import torch

def load_obj(name, directory="."):
    """Load pickled object from file"""
    try:
        with open(os.path.join(directory, name + '.pkl'), 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {name}.pkl in {directory}")
        return None

def prepare_data(data_root, **kwargs):
    """Prepare BraTS data - adapt to your data loading needs"""
    # This should return your processed data
    pass

# 2. Fix the Operator class for segmentation tasks
class Operator:
    def __init__(self, data_loaders, classifier, det_dict, reformer):
        self.test_loader = data_loaders[2]
        self.validation_loader = data_loaders[1]
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer

        # Collect data more memory-efficiently
        print("Collecting 'normal' data...")
        self.normal_data_list = []
        self.normal_labels_list = []
        
        # Process in smaller chunks to avoid OOM
        for batch_data in self.test_loader:
            self.normal_data_list.append(batch_data['image'])
            # For segmentation, create binary tumor presence labels
            has_tumor = (batch_data['mask'].sum(dim=[2,3,4]) > 0).float()
            self.normal_labels_list.append(has_tumor)
        
        self.normal_data_tensor = torch.cat(self.normal_data_list, dim=0)
        self.normal_labels_tensor = torch.cat(self.normal_labels_list, dim=0)

    def operate(self, untrusted_obj):
        """Fixed for segmentation - calculates Dice instead of accuracy"""
        X = untrusted_obj.data
        Y_true = untrusted_obj.labels
        
        batch_size_op = 4  # Smaller batch for 3D data
        results = []
        
        for i in range(0, X.shape[0], batch_size_op):
            try:
                batch_X = X[i:i + batch_size_op]
                batch_Y_true = Y_true[i:i + batch_size_op]
                
                # Reform the input
                batch_X_prime = self.reformer.heal(batch_X)
                
                # Get segmentation predictions
                pred_orig = self.classifier.classify(batch_X, option="prob")
                pred_reformed = self.classifier.classify(batch_X_prime, option="prob")
                
                # For segmentation, compare using Dice score > threshold
                dice_threshold = 0.5
                
                # Calculate Dice scores (simplified - you'd use proper Dice calculation)
                orig_good = self._calculate_dice_batch(pred_orig, batch_Y_true) > dice_threshold
                reformed_good = self._calculate_dice_batch(pred_reformed, batch_Y_true) > dice_threshold
                
                results.extend(list(zip(orig_good, reformed_good)))
                
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA OOM at batch {i}, skipping...")
                continue
                
        return np.array(results)
    
    def _calculate_dice_batch(self, pred, true):
        """Calculate Dice coefficient for batch"""
        # This is a simplified version - implement proper Dice calculation
        pred_binary = (pred > 0.5).float()
        intersection = (pred_binary * true).sum(dim=[1,2,3])
        union = pred_binary.sum(dim=[1,2,3]) + true.sum(dim=[1,2,3])
        dice = (2 * intersection) / (union + 1e-6)
        return dice.cpu().numpy()

class AttackData:
    def __init__(self, examples, labels, name="", directory="."): # Directory argument added
        """
        Wrapper for input data (normal or adversarial).

        examples: Tensor or path to saved object.
        labels: Tensor or numpy array of ground truth labels (for classification).
        name: Identifier string.
        directory: Directory to load examples from if examples is a string path.
        """
        if isinstance(examples, str):
            # This 'utils.load_obj' needs to be compatible with BraTS data (NCHW)
            self.data = utils.load_obj(examples, directory)
            if isinstance(self.data, np.ndarray):
                self.data = torch.tensor(self.data, dtype=torch.float32)
            # Ensure loaded data is NCHW (N, 4, 240, 240)
            if self.data.ndim == 3: # If loaded as CHW, add batch dim
                self.data = self.data.unsqueeze(0)
            # If loaded as NHWC, permute to NCHW
            # if self.data.ndim == 4 and self.data.shape[1:] == torch.Size([BRATS_IMG_SIZE, BRATS_IMG_SIZE, BRATS_CHANNELS]):
            #     self.data = self.data.permute(0, 3, 1, 2)
        else:
            self.data = examples if torch.is_tensor(examples) else torch.tensor(examples, dtype=torch.float32)
            # Ensure tensor input is NCHW
            if self.data.ndim == 3: # If CHW, add batch dim
                self.data = self.data.unsqueeze(0)
            # if self.data.ndim == 4 and self.data.shape[1:] == torch.Size([BRATS_IMG_SIZE, BRATS_IMG_SIZE, BRATS_CHANNELS]):
            #     self.data = self.data.permute(0, 3, 1, 2)


        # Labels should be 1D tensor of class indices for classification accuracy
        self.labels = labels if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
        self.name = name

    def print(self):
        return "Attack:" + self.name


class Evaluator:
    def __init__(self, operator, untrusted_data_obj, graph_dir="./graph"):
        """
        Evaluator for analyzing the defense strategy.

        operator: Operator object.
        untrusted_data_obj: Adversarial or noisy test dataset wrapped in AttackData.
        graph_dir: Path to save graphs.
        """
        self.operator = operator
        self.untrusted_data_obj = untrusted_data_obj # Renamed from untrusted_data for clarity
        self.graph_dir = graph_dir
        self.data_package = operator.operate(untrusted_data_obj) # Result of operating on untrusted data

    def bind_operator(self, operator):
        """
        Replace current operator and re-evaluate.
        """
        self.operator = operator
        self.data_package = self.operator.operate(self.untrusted_data_obj)

    def load_data(self, data_obj):
        """
        Replace current untrusted data and re-evaluate.
        data_obj: An instance of AttackData.
        """
        self.untrusted_data_obj = data_obj
        self.data_package = self.operator.operate(self.untrusted_data_obj)

    def get_normal_acc(self, normal_all_pass_indices):
        """
        Measure classification accuracy on clean data after filtering.
        NOTE: This still assumes a classification accuracy metric (Y_pred == Y_true).
              For BraTS segmentation, 'accuracy' should be replaced by Dice/IoU/Hausdorff.
              You'll need to adapt this if your classifier does full segmentation.

        Returns:
        - both_acc: Accuracy when both detector and reformer pass.
        - det_only_acc: Accuracy with just detector.
        - ref_only_acc: Accuracy with just reformer.
        - none_acc: Accuracy without any defense.
        """
        normal_tups = self.operator.normal_operated_result # Use the pre-calculated normal results
        num_normal = len(normal_tups)

        if num_normal == 0:
            return 0.0, 0.0, 0.0, 0.0 # Handle empty case

        filtered_normal_tups = normal_tups[normal_all_pass_indices]

        # Calculate accuracy for original and reformed samples among *filtered* data
        # 'both_acc' and 'det_only_acc' are based on the filtered subset's performance
        # divided by the *total number of normal samples* (effectively a recall-like metric for "correctly classified & passed filter")
        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC) / num_normal
        det_only_acc = sum(1 for XC, _ in filtered_normal_tups if XC) / num_normal

        # 'ref_only_acc' and 'none_acc' are based on the *entire* original normal set
        ref_only_acc = sum(1 for _, XpC in normal_tups if XpC) / num_normal
        none_acc = sum(1 for XC, _ in normal_tups if XC) / num_normal

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass_indices):
        """
        Measure classification accuracy on adversarial data.
        NOTE: Similar to get_normal_acc, this assumes classification accuracy.
              Adapt for segmentation metrics if needed.
        Returns same metrics as get_normal_acc.
        """
        attack_tups = self.data_package
        num_untrusted = len(attack_tups)

        if num_untrusted == 0:
            return 0.0, 0.0, 0.0, 0.0 # Handle empty case

        filtered_attack_tups = attack_tups[attack_pass_indices]

        # Note: The original logic for `get_attack_acc` has `1 - sum(...)` for both_acc and det_only_acc.
        # This implies it's calculating *error rate* and then converting to accuracy (1 - error_rate).
        # Let's adjust to directly calculate accuracy:
        both_acc = sum(1 for _, XpC in filtered_attack_tups if XpC) / num_untrusted
        det_only_acc = sum(1 for XC, _ in filtered_attack_tups if XC) / num_untrusted

        # These are based on the entire untrusted set, not just filtered.
        ref_only_acc = sum(1 for _, XpC in attack_tups if XpC) / num_untrusted
        none_acc = sum(1 for XC, _ in attack_tups if XC) / num_untrusted

        return both_acc, det_only_acc, ref_only_acc, none_acc


    def plot_various_confidences(self, graph_name, drop_rate,
                                 idx_file="example_idx", # Assuming this is an index file for attacks
                                 confs=(0.0, 10.0, 20.0, 30.0, 40.0),
                                 # This lambda needs to construct the path to your adversarial BraTS data
                                 get_attack_data_name=lambda c: f"brats_adv_carlini_{c}",
                                 data_dir="/kaggle/input/required8"): # Path to load attack data

        """
        Plots performance of the defense under Carlini attacks with varying confidence.
        Assumes adversarial examples are saved in a format loadable by utils.load_obj
        and that they correspond to images and labels of the BraTS type.
        """
        pylab.rcParams['figure.figsize'] = 6, 4
        fig = plt.figure()

        # Placeholders for storing accuracies to plot lines later
        confs_list = []
        both_accs = []
        det_only_accs = []
        ref_only_accs = []
        none_accs = []

        # Loop over each confidence level
        for conf in confs:
            attack_data_path = get_attack_data_name(conf)
            attack_indices_path = os.path.join(data_dir, idx_file)

            try:
                # utils.load_obj must be able to load your BraTS adv data (e.g., .pt, .npy)
                X_adv_all = utils.load_obj(attack_data_path, directory=data_dir)
                # Ensure it's a tensor and NCHW
                if isinstance(X_adv_all, np.ndarray):
                    X_adv_all = torch.from_numpy(X_adv_all).float()
                if X_adv_all.ndim == 3: # Assuming CHW, add batch dim
                    X_adv_all = X_adv_all.unsqueeze(0)
                # if X_adv_all.ndim == 4 and X_adv_all.shape[1:] == torch.Size([BRATS_IMG_SIZE, BRATS_IMG_SIZE, BRATS_CHANNELS]):
                #     X_adv_all = X_adv_all.permute(0, 3, 1, 2) # Convert from NHWC if applicable

                # attack_idx file should contain indices of the original test set that were attacked
                attacked_original_indices = utils.load_obj(idx_file, directory=data_dir)
                if isinstance(attacked_original_indices, np.ndarray):
                    attacked_original_indices = torch.from_numpy(attacked_original_indices).long()

                # Filter attack_idx to be within the bounds of your local test labels
                valid_attack_indices = attacked_original_indices[attacked_original_indices < len(self.operator.normal_labels_tensor)]
                X_adv_filtered = X_adv_all[valid_attack_indices] # Select adversarial examples corresponding to valid indices
                Y_true_filtered = self.operator.normal_labels_tensor[valid_attack_indices] # Get true labels for attacked samples

                attack_dataset = AttackData(X_adv_filtered, Y_true_filtered, name=f"Conf={conf}")

            except FileNotFoundError as e:
                print(f"Warning: Attack data or index file not found for confidence {conf}: {e}. Skipping this confidence level.")
                continue
            except Exception as e:
                print(f"Error loading or processing attack data for confidence {conf}: {e}. Skipping.")
                continue

            self.load_data(attack_dataset) # This re-runs operator.operate() internally
            print(f"Confidence {conf} - # attack samples: {len(attack_dataset.data)}")

            # Use the detector thresholds calculated from validation set (dynamic)
            # You might need to adjust these or pass them more granularly if you want fixed ones
            # from a configuration. Here, drop_rate is used to calculate thresholds from val set.
            thrs = self.operator.get_thrs(drop_rate)

            attack_pass, _ = self.operator.filter(attack_dataset.data, thrs)
            print(f"Passing samples: {len(attack_pass)} out of {len(attack_dataset.data)}")
            accs = self.get_attack_acc(attack_pass)

            # Store results for plotting
            confs_list.append(conf)
            both_accs.append(accs[0])
            det_only_accs.append(accs[1])
            ref_only_accs.append(accs[2])
            none_accs.append(accs[3])

        # Plot lines after collecting all data points
        plt.plot(confs_list, both_accs, 'b-o', label="Both Defense")
        plt.plot(confs_list, det_only_accs, 'g-o', label="Detector Only")
        plt.plot(confs_list, ref_only_accs, 'r-o', label="Reformer Only")
        plt.plot(confs_list, none_accs, 'k-o', label="No Defense")


        plt.xlabel("Attack Confidence")
        plt.ylabel("Classification Accuracy") # Or "Segmentation Metric" if adapted
        plt.title("Defense Performance vs. Attack Confidence (BraTS)")
        plt.legend()
        plt.grid(True)
        graph_path = os.path.join(self.graph_dir, graph_name + ".png")
        os.makedirs(self.graph_dir, exist_ok=True)
        plt.savefig(graph_path)
        plt.close(fig)