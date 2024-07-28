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


class Attention(nn.Module):
    def __init__(self, hist_bins = None) -> None:
        super().__init__()
        self.hist_bins = hist_bins
    def forward(self, q, k, v, mask = None) -> torch.Tensor:
        if True:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                args = [x.contiguous() for x in [q, k, v, mask] if x is not None] # .contiguous().half()
                return F.scaled_dot_product_attention(*args).to(q.dtype)
                pass
        else:
            s = q.shape[-1] ** -0.5
            attn = F.softmax(torch.einsum('...id,...jd->...ij', q, k) * s, -1)
            torch.histogram(attn)
            return torch.einsum('...ij,...jd->...id', attn, v)

class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3*embed_dim, bias=bias)
        self.inner_attn = Attention()
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.LayerNorm(2*embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

    def _forward(self, x: torch.Tensor,
                 mask = None):
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        context = self.inner_attn(q, k, v, mask = mask)
        message = self.out_proj(
            context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

    def forward(self, x0, x1, mask0=None, mask1=None):
        return self._forward(x0, mask = mask0), self._forward(x1, mask = mask1)


class DenseRefiner(nn.Module):
    def __init__(self, conf, **kwargs) -> None:
        super().__init__(**kwargs)
        h, n, d = conf.refiner_num_heads, conf.refiner_n_layers, conf.refiner_descriptor_dim
        self.num_heads = h
        hidden_dim = (2) * conf.refiner_proj_dim
        self.self_attn = nn.ModuleList(
            [Transformer(hidden_dim, h, conf.flash) for _ in range(n)])
        self.project_descriptor = nn.Linear(d, conf.refiner_proj_dim)
        self.to_scores = nn.Linear(hidden_dim, 1)
        self.to_matchability = nn.Linear(hidden_dim, 1)
        self.conf = conf
    
    def forward(self, data):
        desc0, desc1 = data['node_knn_feats_A'], data['node_knn_feats_B']
        desc0 = self.project_descriptor(desc0)
        desc1 = self.project_descriptor(desc1)
        B,N,D = desc0.shape
        stacked_feature_maps0 = torch.cat((desc0.reshape(B,1,N,D).expand((B,N,N,D)), desc1.reshape(B,1,N,D).expand((B,N,N,D))),dim=-1).reshape(B, N**2, 2*D)
        stacked_feature_maps1 = torch.cat((desc1.reshape(B,1,N,D).expand((B,N,N,D)), desc0.reshape(B,1,N,D).expand((B,N,N,D))),dim=-1).reshape(B, N**2, 2*D)
        mask0, mask1 = None, None

        for i in range(self.conf.n_layers):
            stacked_feature_maps0, stacked_feature_maps1 = self.self_attn[i](stacked_feature_maps0, stacked_feature_maps1, mask0 = mask0, mask1 = mask1)
        scores0, scores1 = self.to_scores(stacked_feature_maps0), self.to_scores(stacked_feature_maps1)
        symmetric_scores = scores0.reshape(B,N,N)# + scores1.mT
        log_scores = torch.log_softmax(symmetric_scores, dim = -1) + torch.log_softmax(symmetric_scores, dim = -2)
        matchability0, matchability1 = self.to_matchability(stacked_feature_maps0).reshape(B,N,N).max(dim=-1).values, self.to_matchability(stacked_feature_maps1).reshape(B,N,N).max(dim=-2).values
        data['refiner_scores'] = [log_scores]
        data['refiner_matchability'] = [(matchability0, matchability1)]
        return data
