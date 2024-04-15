import numpy as np
import torch
import torch.nn as nn
from .model import sfmregger
from omnireg.models.simtr_wrapper import SimTr
import pycolmap
from omnireg.utils import extract_pcd_from_colmap_model

class SfMReger(nn.Module):
    def __init__(self, weights_path, mode = "se3", **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = SimTr(sfmregger(weights_path))
        self.mode = mode
    
    def register_reconstructions(self, model_A: pycolmap.Reconstruction, model_B: pycolmap.Reconstruction):
        pcd_A, view_A, rgb_A = extract_pcd_from_colmap_model(model_A)
        pcd_B, view_B, rgb_B = extract_pcd_from_colmap_model(model_B)
        return self.register_pointclouds(pcd_A, view_A, pcd_B, view_B)

    def register_pointclouds(self, pcd_A: np.ndarray, view_A: np.ndarray, pcd_B: np.ndarray, view_B: np.ndarray):
        shared_scale = False if self.mode == "sim3" else True
        results = self.model.register(pcd_A, pcd_B, viewpoints_A=view_A, viewpoints_B=view_B, shared_scale=shared_scale)
        return results