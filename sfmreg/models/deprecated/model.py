from pathlib import Path
from types import SimpleNamespace
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Callable
from time import perf_counter
import open3d as o3d
from easydict import EasyDict as edict

from .point_learners.ppf_transformer import PPFWrapper
from .point_learners.roitr_backbone import RIPointTransformerWrapper
from .global_matcher.matcher import GlobalMatcher
from .refinement.refiner import Refiner
from .refinement.optimal_transport import LearnableLogOptimalTransport
from .refinement.dense_refiner import DenseRefiner
from sfmreg.utils import get_correspondences, estimate_normals
from sfmreg.solvers import SVDSOPPSolver

class OmniGlue(nn.Module):
    default_conf = edict(
        name = 'omniglue',  # just for interfacing
        descriptor_dim = 256,
        n_layers = 4,
        num_heads = 1,
        flash = True,  # enable FlashAttention if available.
        mp = False,  # enable mixed precision
        weights = None,
        solver = SVDSOPPSolver(),
    )

    required_data_keys = [
        'pc_A', 'pc_B']
    backbones = {"omniglue": PPFWrapper, "roitr": RIPointTransformerWrapper}
    refiners = {"local_softmax": Refiner, "OT": LearnableLogOptimalTransport, "dense": DenseRefiner}
    version = "v0.0"

    def __init__(self, conf) -> None:
        super().__init__()
        self.default_conf.update(conf)
        self.conf = conf = self.default_conf
        self.global_matcher = GlobalMatcher(conf)
        self.refiner = OmniGlue.refiners[conf.refiner](conf) if conf.use_refiner else None
        self.point_learner = OmniGlue.backbones[conf.backbone](conf)
        self.solver = conf.solver
        self.normalize_features = conf.normalize_features
        self.signature = conf.signature
    
    def forward(self, data: dict, score_threshold = -3) -> dict:
        """
        Match keypoints and descriptors between two images

        Input (dict):
            keypoints_A
                [B x M//16 x 3]
            pc_A: dict
                keypoints: [B x M x 3]
            pc_B: dict
                keypoints: [B x N x 3]
            keypoints_B
                [B x N//16 x 3]

        Output (dict):
            log_assignment: [B x M+1 x N+1]
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]], scores: List[[Si]]
        """
        with torch.autocast(enabled=self.conf.mp, device_type='cuda'):
            return self._forward(data, score_threshold = score_threshold)
    
    def matches_to_ragged(self, data):
        ragged_matches_A = []
        ragged_matches_B = []
        for b in range(data['batch_size']):
            matches_b = data['matches'][0] == b
            ragged_match_A_b = data['matches'][1][matches_b] + (data["offset_coarse_A"][b-1] if b > 0 else 0)
            ragged_match_B_b = data['matches'][2][matches_b] + (data["offset_coarse_B"][b-1] if b > 0 else 0)
            ragged_matches_A.append(ragged_match_A_b)
            ragged_matches_B.append(ragged_match_B_b)
        data['ragged_matches'] = torch.cat(ragged_matches_A), torch.cat(ragged_matches_B)
    
    def _forward(self, data: dict, score_threshold = -3) -> dict:
        # Extract features 
        data = self.point_learner(data)

        # Normalize features
        if self.normalize_features:
            data["descriptor_A"] = F.normalize(data["descriptor_A"], p=2, dim=-1)
            data["descriptor_B"] = F.normalize(data["descriptor_B"], p=2, dim=-1)
        
            
        # Following steps require non-ragged tensors
        if "is_ragged" in data:
            from sfmreg.utils import pad_to_length_ragged
            padded_descriptor_A, mask_A = pad_to_length_ragged(data["descriptor_A"], data["num_coarse_A"], max(data["num_coarse_A"]))
            padded_keypoints_A, mask_A = pad_to_length_ragged(data["keypoints_A"], data["num_coarse_A"], max(data["num_coarse_A"]))
            padded_descriptor_B, mask_B = pad_to_length_ragged(data["descriptor_B"], data["num_coarse_B"], max(data["num_coarse_B"]))
            padded_keypoints_B, mask_B = pad_to_length_ragged(data["keypoints_B"], data["num_coarse_B"], max(data["num_coarse_B"]))

            data["padded_descriptor_A"] = padded_descriptor_A
            data["padded_descriptor_B"] = padded_descriptor_B
            data["padded_keypoints_A"] = padded_keypoints_A
            data["padded_keypoints_B"] = padded_keypoints_B            
            data["pad_mask_A"] = mask_A 
            data["pad_mask_B"] = mask_B
        
        # Match coarse features
        data = self.global_matcher(data)
        scores = data['scores'] 
        data['matches'], data['global_confidence'] = self.extract_coarse(scores, all_matchability = data['matchability'], mode = "avg")
        
        self.matches_to_ragged(data)
        data['coarse_src_corr_pts'] = data['keypoints_A'][data['ragged_matches'][0]]
        data['coarse_tgt_corr_pts'] = data['keypoints_B'][data['ragged_matches'][1]]
        data['coarse_corr_batch_ids'] = data['matches'][0]
        
        # Refine using fine features
        if self.refiner is not None:
            # 2. get ground truth node correspondences
            data = self.assign_neighbourhoods(data)
            data = self.refiner(data)
            data['refiner_matches'], data['refiner_confidence'] = self.extract_fine(data['refiner_scores'], all_matchability = data["refiner_matchability"], score_threshold = score_threshold, mode = "avg")
            data['refiner_points_A'] = data['node_knn_points_A']
            data['refiner_points_B'] = data['node_knn_points_B']

        data['src_corr_pts'] = data['refiner_points_A'][data['refiner_matches'][0], data['refiner_matches'][1]]
        data['tgt_corr_pts'] = data['refiner_points_B'][data['refiner_matches'][0], data['refiner_matches'][2]]
        data['corr_batch_ids'] = data['fine_knn_batch_ids'][data['refiner_matches'][0]]
        data["confidence"] = data['refiner_confidence']
        data["signature"] = self.signature
        return data
    
    @torch.no_grad()    
    def assign_neighbourhoods(self, data):
        from ..pointops import knnquery
        inds_A, dists_A = knnquery(self.conf.refine_num_nn, data["fine_points_A"], data["keypoints_A"], data["offset_fine_A"], data["offset_coarse_A"])
        inds_B, dists_B = knnquery(self.conf.refine_num_nn, data["fine_points_B"], data["keypoints_B"], data["offset_fine_B"], data["offset_coarse_B"])
        data['node_knn_feats_A'] = data["fine_features_A"][inds_A[data["ragged_matches"][0]]]
        data['node_knn_feats_B'] = data["fine_features_B"][inds_B[data["ragged_matches"][1]]]
        data['node_knn_points_A'] = data["fine_points_A"][inds_A[data["ragged_matches"][0]]]
        data['node_knn_points_B'] = data["fine_points_B"][inds_B[data["ragged_matches"][1]]]
        data['fine_knn_batch_ids'] = data["batch_ids_fine_A"][inds_A[:,0][data["ragged_matches"][0]]] # [:,0] because batch will be same for all knn grps
        return data

    @torch.no_grad()
    def extract_coarse(self, all_scores, all_matchability = None, score_threshold = -3, mode = "avg"):
        if mode == "avg":
            avg_score = sum(all_scores) / len(all_scores)
            scores = avg_score
        elif mode == "final":
            scores = all_scores[-1]
        if all_matchability is not None:
            matchability_A = all_matchability[-1][0]
            matchability_B = all_matchability[-1][1]
            scores = scores + matchability_A[...,None] + matchability_B[...,None,:]

        matches = torch.nonzero((scores == scores.max(dim=-1, keepdim = True).values) * (scores == scores.max(dim=-2, keepdim = True).values) * (scores >= score_threshold), as_tuple=True)
        confidence = F.sigmoid(scores[matches[0], matches[1], matches[2]])
        return matches, confidence

    @torch.no_grad()
    def extract_fine(self, all_scores, all_matchability = None, score_threshold = -3, mode = "final"):
        if mode == "avg":
            avg_score = sum(all_scores) / len(all_scores)
            scores = avg_score
        elif mode == "final":
            scores = all_scores[-1]
        if all_matchability[-1][0] is not None:
            matchability_A = all_matchability[-1][0]
            matchability_B = all_matchability[-1][1]
            scores = scores + matchability_A[...,None] + matchability_B[...,None,:]

        matches = torch.nonzero((scores == scores.max(dim=-1, keepdim = True).values) * (scores == scores.max(dim=-2, keepdim = True).values) * (scores >= score_threshold), as_tuple=True)
        confidence = F.sigmoid(scores[matches[0], matches[1], matches[2]])
        return matches, confidence

    @torch.inference_mode()
    def register(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None):
        self.train(False)
        matches = self.match(pointcloud_A, pointcloud_B, normals_A = normals_A, normals_B = normals_B)
        self.solver.solve(matches, "scale+rot+translation")   
    
    @torch.inference_mode()
    def match(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None, score_threshold = -3):
        pointcloud_A, pointcloud_B = pointcloud_A.float(), pointcloud_B.float()
        self.train(False)
        if normals_A is None or normals_B is None:
            normals_A = estimate_normals(pointcloud_A).float()
            normals_B = estimate_normals(pointcloud_B).float()
        data = {"pc_A": pointcloud_A, "pc_B": pointcloud_B, 
                "normals_A": normals_A, "normals_B": normals_B,
                "num_pt_A": torch.tensor(pointcloud_A.shape[:1], device = pointcloud_A.device),
                "num_pt_B": torch.tensor(pointcloud_B.shape[:1], device = pointcloud_B.device),
                "is_ragged": True,
                "batch_size": 1,
                }
        data = self(data, score_threshold = score_threshold)
        #Below needed for Predator eval compat
        return data