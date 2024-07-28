import torch.nn as nn
import torch

class WeightedCombination(nn.Module):
    def __init__(self, losses, weights) -> None:
        super().__init__()
        self.losses = losses
        self.weights = weights
    
    def forward(self, data):
        return sum(loss(data) * weight for loss, weight in zip(self.losses, self.weights))