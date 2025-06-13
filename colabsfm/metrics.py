import torch
from .utils import from_homogeneous, to_homogeneous

@torch.no_grad()
def compute_inlier_ratio(p_hat, q_hat, transform, threshold_setting = "tdmatch", threshold = None):
    if isinstance(p_hat, list):
        return [compute_inlier_ratio(p_hat[t], q_hat[t], transform[t], threshold_setting = threshold_setting) for t in range(len(p_hat))]
    # (N,D)
    if threshold_setting == "tdmatch":
        threshold = 0.1
    elif threshold_setting == "fdmatch":
        threshold = 0.04
    N,D = p_hat.shape
    if N == 0:
        return 0.
    p_hat_to_q = from_homogeneous((transform @ to_homogeneous(p_hat).mT).mT)
    distance = (p_hat_to_q - q_hat).norm(dim=-1)
    IR = (distance < threshold).float().mean().item()
    return IR

@torch.no_grad()
def compute_mnn_inlier_ratio(p_hat, q_hat, transform):
    if isinstance(p_hat, list):
        return [compute_inlier_ratio(p_hat[t], q_hat[t], transform[t]) for t in range(len(p_hat))]
    # (N,D)
    N,D = p_hat.shape
    if N == 0:
        return 0
    p_hat_to_q = from_homogeneous((transform @ to_homogeneous(p_hat).mT).mT)
    distance = (p_hat_to_q - q_hat).norm(dim=-1)
    mnn = torch.nonzero((distance == distance.min(dim=0)) * (distance == distance.min(dim=1)))
    IR = len(mnn)/len(p_hat)
    return IR

@torch.no_grad()
def compute_feature_matching_recall(IRs):
    FMR = (IRs > 0.05).float().mean().item()
    return FMR