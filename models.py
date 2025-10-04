# models.py
import torch.nn as nn

class DiabetesNet(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # output = raw logit
        )

    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x).squeeze(-1)  # returns shape (batch,)
