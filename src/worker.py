import sys
sys.path.append('/kaggle/input/required8')
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
from src.utils import prepare_data
import src.utils as utils
import matplotlib.pyplot as plt

class AEDetector:
    def __init__(self, model_class, model_path, p=1, device=None,model_kwargs=None):
        """
        Error-based detector.

        model_class: The class of the AE model (e.g., DenoisingAutoEncoder).
        model_path: Path to the model's saved state_dict (.pth).
        p: Power for error norm (e.g., 1 = L1, 2 = L2).
        device: torch.device object or string ("cuda" / "cpu").
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.p = p
        
    def mark(self, X):
        """
        X: A PyTorch tensor of shape [N, C, H, W] on CPU or GPU.
        Returns: 1D numpy array of anomaly scores (reconstruction errors).
        """
        self.model.eval()

        with torch.no_grad():
            if X.ndim == 4 and X.shape[1:] == torch.Size([28, 28, 1]):
                
                X = X.permute(0, 3, 1, 2)

            X = X.to(self.device)
            recon = self.model(X)
            diff = torch.abs(X - recon)  # Absolute error
            score = torch.mean(torch.pow(diff, self.p), dim=[1, 2, 3])  # Per-sample score
            return score.cpu().numpy()  # Convert to NumPy for compatibility

    def print(self):
        return "AEDetector:" + os.path.basename(self.path)


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
        path: Path to the saved model (.pth).
        device: torch.device or str ("cuda" or "cpu").
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.image_shape = self.model.image_shape

    def heal(self, X):
        """
        X: Tensor [N, C, H, W] (should be on same device as model)
        Returns: Reconstructed input clipped between 0 and 1 (as Torch Tensor)
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            if X.ndim == 3:  # Single image
                if X.shape[-1] == 1:  # HWC format
                    X = X.permute(2, 0, 1).unsqueeze(0)  # Convert to NCHW
                else:  # CHW format
                    X = X.unsqueeze(0)  # Add batch dimension
            elif X.ndim == 4:  # Batch of images
                if X.shape[-1] == 1:  # NHWC format
                    X = X.permute(0, 3, 1, 2)  # Convert to NCHW
        
        # Ensure proper normalization (autoencoders are sensitive to input range)
            if X.max() > 1.0:
                X = X / 255.0
            print(f"Input range: [{X.min():.6f}, {X.max():.6f}]")
            print(f"Input mean: {X.mean():.6f}")
            print(f"Input std: {X.std():.6f}")
        
        # Check intermediate representations
            if hasattr(self.model, 'encoder'):
                latent = self.model.encoder(X)
                print(f"Latent range: [{latent.min():.6f}, {latent.max():.6f}]")
                print(f"Latent mean: {latent.mean():.6f}")
                print(f"Latent near-zero count: {(torch.abs(latent) < 1e-6).sum().item()}/{latent.numel()}")
            recon = self.model(X)
            recon = torch.clamp(recon, 0.0, 1.0)
            print(f"Output range: [{recon.min():.6f}, {recon.max():.6f}]")
            print(f"Output mean: {recon.mean():.6f}")
            return recon

    def print(self):
        return "SimpleReformer:" + os.path.basename(self.path)


def JSD(P, Q):
    """
    Jensen-Shannon Divergence between two 1D arrays (P, Q).
    P, Q: Numpy arrays representing discrete distributions.
    """
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
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
        Xp = self.prober.heal(X)
        Xr = self.reconstructor.heal(X)

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
        model_kwargs: dict of kwargs for model_class constructor
        """
        self.path = classifier_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_class(**(model_kwargs or {})).to(self.device)
        self.model.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.model.eval()

    def classify(self, X, option="logit", T=1):
        """
        Classify input.

        X: Torch tensor [N, C, H, W]
        option: "logit" to return raw logits, "prob" to return softmax probs.
        T: Temperature (used only with option="prob")
        Returns: NumPy array
        """
        if X.ndim == 4 and X.shape[1:] == torch.Size([28, 28, 1]):
            X = X.permute(0, 3, 1, 2)
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()

            X = X.to(self.device)
            logits = self.model(X)

            if option == "logit":
                return logits.cpu().numpy()

            elif option == "prob":
                probs = F.softmax(logits / T, dim=1)
                return probs.cpu().numpy()

            else:
                raise ValueError(f"Invalid option: {option}. Use 'logit' or 'prob'.")

    def print(self):
        return "Classifier:" + os.path.basename(self.path)


class Operator:
    def __init__(self, data, classifier, det_dict, reformer):
        """
        Operator that wraps the data, classifier, detector(s), and reformer logic.

        data: MNIST object with .train_loader, .validation_loader, .test_loader
        classifier: Classifier object (must have .classify(X) method)
        det_dict: Dictionary of detectors, each with .mark(X)
        reformer: Reformer object (must have .heal(X))
        """
        self.data = data
        self.classifier = classifier
        self.det_dict = det_dict
        self.reformer = reformer

        test_imgs = torch.stack([img for img, _ in data.test_data])
        test_labels = torch.tensor([label for _, label in data.test_data])
        self.normal = self.operate(AttackData(test_imgs, test_labels, "Normal"))

    def get_thrs(self, drop_rate):
        """
        Calculates thresholds for filtering from validation set.
        drop_rate: Dict mapping detector names to float drop rates.
        """
        thrs = {}
        val_imgs = torch.stack([img for img, _ in self.data.validation_data])
        for name, detector in self.det_dict.items():
            num = int(len(val_imgs) * drop_rate[name])
            marks = detector.mark(val_imgs)
            sorted_marks = np.sort(marks)
            thrs[name] = sorted_marks[-num]
        return thrs

    def operate(self, untrusted_obj):
        """
        Classifies original and reformed inputs using the classifier.
        Returns: Array of (original_correct, reformed_correct) pairs.
        """
        X = untrusted_obj.data
        Y_true = untrusted_obj.labels

        with torch.no_grad():
            X_prime = self.reformer.heal(X)
            Y_pred = torch.tensor(np.argmax(self.classifier.classify(X), axis=1))
            Y_judgement = (Y_pred == Y_true[:len(X_prime)])

            Yp_pred = torch.tensor(np.argmax(self.classifier.classify(X_prime), axis=1))
            Yp_judgement = (Yp_pred == Y_true[:len(X_prime)])

        return np.array(list(zip(Y_judgement.cpu().numpy(), Yp_judgement.cpu().numpy())))

    def filter(self, X, thrs):
        """
        Filters inputs using detector thresholds.
        Returns indices that passed all filters.
        """
        all_pass = np.arange(X.shape[0])
        collector = {}

        for name, detector in self.det_dict.items():
            marks = detector.mark(X)
            idx_pass = np.argwhere(marks < thrs[name]).flatten()
            collector[name] = len(idx_pass)
            all_pass = np.intersect1d(all_pass, idx_pass)

        return all_pass, collector

    def print(self):
        components = [self.reformer, self.classifier]
        return " ".join(obj.print() for obj in components)

class AttackData:
    def __init__(self, examples, labels, name="",directory="/kaggle/input/required8"):
        """
        Wrapper for input data (normal or adversarial).
        
        examples: Tensor or path to saved object.
        labels: Tensor or numpy array of ground truth labels.
        name: Identifier string.
        """
        if isinstance(examples, str):
            self.data = utils.load_obj(examples,directory)  # Should return a torch tensor or NumPy array
            if isinstance(self.data, np.ndarray):
                self.data = torch.tensor(self.data, dtype=torch.float32)
        else:
            self.data = examples if torch.is_tensor(examples) else torch.tensor(examples, dtype=torch.float32)

        self.labels = labels if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
        self.name = name

    def print(self):
        return "Attack:" + self.name


class Evaluator:
    def __init__(self, operator, untrusted_data, graph_dir="./graph"):
        """
        Evaluator for analyzing the defense strategy.

        operator: Operator object.
        untrusted_data: Adversarial or noisy test dataset wrapped in AttackData.
        graph_dir: Path to save graphs.
        """
        self.operator = operator
        self.untrusted_data = untrusted_data
        self.graph_dir = graph_dir
        self.data_package = operator.operate(untrusted_data)

    def bind_operator(self, operator):
        """
        Replace current operator and re-evaluate.
        """
        self.operator = operator
        self.data_package = operator.operate(self.untrusted_data)

    def load_data(self, data):
        """
        Replace current untrusted data and re-evaluate.
        """
        self.untrusted_data = data
        self.data_package = self.operator.operate(self.untrusted_data)

    def get_normal_acc(self, normal_all_pass):
        """
        Measure classification accuracy on clean data after filtering.

        Returns:
        - both_acc: Accuracy when both detector and reformer pass.
        - det_only_acc: Accuracy with just detector.
        - ref_only_acc: Accuracy with just reformer.
        - none_acc: Accuracy without any defense.
        """
        normal_tups = self.operator.normal
        num_normal = len(normal_tups)
        filtered_normal_tups = normal_tups[normal_all_pass]

        both_acc = sum(1 for _, XpC in filtered_normal_tups if XpC) / num_normal
        det_only_acc = sum(1 for XC, _ in filtered_normal_tups if XC) / num_normal
        ref_only_acc = sum(1 for _, XpC in normal_tups if XpC) / num_normal
        none_acc = sum(1 for XC, _ in normal_tups if XC) / num_normal

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def get_attack_acc(self, attack_pass):
        """
        Measure classification accuracy on adversarial data.

        Returns same metrics as get_normal_acc.
        """
        attack_tups = self.data_package
        num_untrusted = len(attack_tups)
        filtered_attack_tups = attack_tups[attack_pass]

        both_acc = 1 - sum(1 for _, XpC in filtered_attack_tups if not XpC) / num_untrusted
        det_only_acc = 1 - sum(1 for XC, _ in filtered_attack_tups if not XC) / num_untrusted
        ref_only_acc = sum(1 for _, XpC in attack_tups if XpC) / num_untrusted
        none_acc = sum(1 for XC, _ in attack_tups if XC) / num_untrusted

        return both_acc, det_only_acc, ref_only_acc, none_acc

    def plot_various_confidences(self, graph_name, drop_rate,
                                 idx_file="example_idx",
                                 confs=(0.0, 10.0, 20.0, 30.0, 40.0),
                                 get_attack_data_name=lambda c: f"example_carlini_{c}",data_dir="/kaggle/input/required8"):
        """
        Plots performance of the defense under Carlini attacks with varying confidence.
        """
        pylab.rcParams['figure.figsize'] = 6, 4
        fig = plt.figure()

        # Loop over each confidence level
        for conf in confs:
            attack_data_name = get_attack_data_name(conf)
            attack_data = utils.load_obj(attack_data_name,directory = data_dir)  # assumed to return AttackData-compatible tensors
            attack_idx = utils.load_obj(idx_file,directory = data_dir)
            max_idx = len(attack_data) - 1
            attack_idx = attack_idx[attack_idx < len(self.untrusted_data.labels)]
            X_adv = attack_data[attack_idx]
            Y_true = self.untrusted_data.labels[attack_idx]
            attack_dataset = AttackData(X_adv, Y_true, name=f"Conf={conf}")

            self.load_data(attack_dataset)
            print(f"Confidence {conf} - # attack samples: {len(attack_dataset.data)}")
            thrs = {
                "I": 0.85,   # manually chosen threshold for detector_I
                "II": 0.85   # manually chosen threshold for detector_II
            }
            attack_pass, _ = self.operator.filter(attack_dataset.data, thrs)
            print(f"Passing samples: {len(attack_pass)} out of {len(attack_dataset.data)}")
            accs = self.get_attack_acc(attack_pass)

            plt.plot(conf, accs[0], 'bo')  # both_acc
            plt.plot(conf, accs[1], 'go')  # det_only_acc
            plt.plot(conf, accs[2], 'ro')  # ref_only_acc
            plt.plot(conf, accs[3], 'ko')  # none_acc

        plt.xlabel("Attack Confidence")
        plt.ylabel("Classification Accuracy")
        plt.title("Defense Performance vs. Attack Confidence")
        plt.legend(["both", "det_only", "ref_only", "none"])
        plt.grid(True)
        graph_path = os.path.join(self.graph_dir, graph_name + ".png")
        os.makedirs(self.graph_dir, exist_ok=True)
        plt.savefig(graph_path)
        plt.close(fig)
