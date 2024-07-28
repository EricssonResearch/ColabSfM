import torch
from colabsfm.models.model import OmniGlue

def sanity_check_gpu():
    device = "cuda"
    model = OmniGlue().to(device)
    B = 2
    N = 1000
    C = 3
    batch = {"pc_A": torch.randn(B, N, C).to(device), "pc_B": torch.randn(B, N, C).to(device)}
    model(batch)