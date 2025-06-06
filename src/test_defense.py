import sys
sys.path.append('/kaggle/input/required8')
import torch
from dataloader import MNIST
from utils import prepare_data, load_obj
from DAE_model import DenoisingAutoEncoder
from model import CNNClassifier

# ---- Load models ----
detector_I = AEDetector(DenoisingAutoEncoder, "/kaggle/input/required8/MNIST_I.pth", p=2, model_kwargs={'image_shape': (1, 28, 28), 'structure': [3,"average",3], 'v_noise': 0.1, 'activation': 'relu', 'model_dir': './defensive_models/', 'reg_strength': 0.0})
detector_II = AEDetector(DenoisingAutoEncoder, "/kaggle/input/required8/MNIST_II.pth", p=1, model_kwargs={'image_shape': (1, 28, 28), 'structure': [3], 'v_noise': 0.1, 'activation': 'relu', 'model_dir': './defensive_models/', 'reg_strength': 0.0})
reformer = SimpleReformer(DenoisingAutoEncoder, "/kaggle/input/required8/MNIST_I.pth", model_kwargs={'image_shape': (1, 28, 28), 'structure': [3,"average",3], 'v_noise': 0.1, 'activation': 'relu', 'model_dir' : './defensive_models/', 'reg_strength': 0.0})
id_reformer = IdReformer()
classifier = Classifier(CNNClassifier, "/kaggle/input/required8/example_classifier.pth",model_kwargs={'params' : [32,32,64,64,200,200]})

# ---- Compose detector dictionary ----
detector_dict = {
    "I": detector_I,
    "II": detector_II
}

# ---- Load MNIST data ----
dataset = MNIST()
operator = Operator(dataset, classifier, detector_dict, reformer)



# ---- Load adversarial example indices and labels ----
idx = load_obj("example_idx",directory='/kaggle/input/required8')
_, _, Y = prepare_data(dataset, idx)

# ---- Load adversarial examples manually and convert ----
examples_np = load_obj("example_carlini_0.0", directory='/kaggle/input/required8')  # numpy array with shape (N,28,28,1)

# Convert numpy array to torch tensor and permute to (N, C, H, W)
examples = torch.tensor(examples_np, dtype=torch.float32)
if examples.ndim == 4 and examples.shape[-1] == 1:
    examples = examples.permute(0, 3, 1, 2)

# ---- Use AttackData by overriding data and labels ----
test_attack = AttackData("example_carlini_0.0", Y, name="Carlini L2 0.0", directory='/kaggle/input/required8')

# Override the loaded data with our tensor (monkey patch)
test_attack.data = examples

# ---- Create Evaluator and plot performance ----
evaluator = Evaluator(operator, test_attack)
evaluator.plot_various_confidences(
    graph_name="defense_performance",
    drop_rate={"I": 0.1, "II": 0.1}
)
