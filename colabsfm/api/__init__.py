import numpy as np
import torch
import torch.nn as nn
from .model import load_RoITr
# from colabsfm.models.roitr_wrapper import RoITr
import pycolmap
from colabsfm.utils import extract_pcd_from_colmap_model
from colabsfm.benchmarks.benchmark_utils import to_array

class RefineRoITr(nn.Module):
    def __init__(self, *, weights_path = None, mode = "se3", simplified_return_dict = True, **kwargs) -> None:
        super().__init__(**kwargs)
        if weights_path is None and mode == "se3":
            weights_path = "https://github.com/EricssonResearch/ColabSfM/releases/download/weights/sfmreg_finetuned.pth"
        elif weights_path is None and mode == "sim3":
            weights_path = "https://github.com/EricssonResearch/ColabSfM/releases/download/weights/sfmreg_only_sim3.pth"
        
        self.model = load_RoITr(weights_path)
        self.mode = mode
        self.simplified_return_dict = simplified_return_dict
    
    def register_reconstructions(self, model_A: pycolmap.Reconstruction, model_B: pycolmap.Reconstruction):
        pcd_A, view_A, rgb_A = extract_pcd_from_colmap_model(model_A)
        pcd_B, view_B, rgb_B = extract_pcd_from_colmap_model(model_B)
        return self.register_pointclouds(pcd_A, view_A, pcd_B, view_B)

    def register_pointclouds(self, pcd_A: np.ndarray, view_A: np.ndarray, pcd_B: np.ndarray, view_B: np.ndarray):
        shared_scale = False if self.mode == "sim3" else True

        results = self.model.register(
            torch.from_numpy(pcd_A).cuda(), 
            torch.from_numpy(pcd_B).cuda(), 
            viewpoints_A=view_A, 
            viewpoints_B=view_B, 
            shared_scale=shared_scale)
        if self.simplified_return_dict:
            results = {
                "transformation": results["transformation"],
                "num_matches": len(results['src_corr_points']),
                "matching_points_A": to_array(results['src_corr_points']),
                "matching_points_B": to_array(results['tgt_corr_points']),
            }
        return results