import sys
from omnireg.roitr.model.model import *
from omnireg.roitr.configs.utils import load_config
from easydict import EasyDict as edict

import torch.nn as nn
import torch
import numpy as np

class RIPointTransformerWrapper(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        conf = edict(conf)
        #assert conf.descriptor_dim == 256, "Only original dim 256 allowed!"
        config = edict(load_config("third_party/RoITr/configs/test/tdmatch.yaml"))
        config.update(conf)
        self.backbone = RIPointTransformer(transformer_architecture=config.transformer_architecture, with_cross_pos_embed=config.with_cross_pos_embed, 
                                           factor=1, encoder_only = config.encoder_only, use_global_transformer = config.use_global_transformer, 
                                           nsample = config.nsample if 'nsample' in config else None, return_dictionary = True, descriptor_dim = conf.descriptor_dim, ppf_mode = conf.ppf_mode)
        self.config = config
    def forward(self, data):
        sizes_A = data['num_pt_A']
        sizes_B = data['num_pt_B']

        points_A, points_B = data['pc_A'].reshape(-1,3), data['pc_B'].reshape(-1,3)
        normals_A, normals_B = data['normals_A'].reshape(-1,3), data['normals_B'].reshape(-1,3)

        src_pcd = src_raw_pcd = points_A
        tgt_pcd = tgt_raw_pcd = points_B
        src_normals, tgt_normals = normals_A, normals_B
        src_feats, tgt_feats = torch.ones_like(points_A[...,:1]), torch.ones_like(points_B[...,:1]) 
        if not isinstance(sizes_A,torch.Tensor):
            src_o, tgt_o = torch.from_numpy(np.array(sizes_A)).to(src_pcd).int(), torch.from_numpy(np.array(sizes_B)).to(tgt_pcd).int()
        else:
            src_o, tgt_o = sizes_A.int(), sizes_B.int() # geotransformer pointops require int32 not int64
        src_o, tgt_o = torch.cumsum(src_o, 0).int(), torch.cumsum(tgt_o, 0).int()
        feature_pyramid = \
                self.backbone([src_raw_pcd, src_feats, src_o, src_normals], [tgt_pcd, tgt_feats, tgt_o, tgt_normals], src_pcd)

        data['keypoints_A'], data['keypoints_B'] = feature_pyramid.s_p4, feature_pyramid.t_p4
        data['descriptor_A'], data['descriptor_B'] = feature_pyramid.s_g_x4, feature_pyramid.t_g_x4
        data['offset_coarse_A'] = feature_pyramid.s_o4
        data['num_coarse_A'] = torch.diff(torch.cat((torch.zeros_like(feature_pyramid.s_o4[:1]),feature_pyramid.s_o4)))
        
        data['offset_coarse_B'] = feature_pyramid.t_o4
        data['num_coarse_B'] = torch.diff(torch.cat((torch.zeros_like(feature_pyramid.t_o4[:1]),feature_pyramid.t_o4)))

        data['offset_fine_A'] = feature_pyramid[f's_o{self.config.refine_level}']
        data['fine_features_A'] = feature_pyramid[f's_x{self.config.refine_level}']
        data['fine_points_A'] = feature_pyramid[f's_p{self.config.refine_level}']
        data['num_fine_A'] = torch.diff(data['offset_fine_A'], prepend = torch.zeros_like(data['offset_fine_A'][:1]))
        data['batch_ids_fine_A'] = torch.cat([idx * torch.ones(num, device = "cuda").int() for idx, num in enumerate(data['num_fine_A'])])
        
        data['offset_fine_B'] = feature_pyramid[f't_o{self.config.refine_level}']
        data['fine_features_B'] = feature_pyramid[f't_x{self.config.refine_level}']
        data['fine_points_B'] = feature_pyramid[f't_p{self.config.refine_level}']
        data['num_fine_B'] = torch.diff(data['offset_fine_B'], prepend = torch.zeros_like(data['offset_fine_B'][:1]))
        data['batch_ids_fine_B'] = torch.cat([idx * torch.ones(num, device = "cuda").int() for idx, num in enumerate(data['num_fine_B'])])

            
        return data
