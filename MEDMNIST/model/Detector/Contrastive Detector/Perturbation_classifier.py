import torch
import torch.nn as nn

class ContrastiveClassifier(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Input: embedding_dim instead of 3x28x28 images
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


