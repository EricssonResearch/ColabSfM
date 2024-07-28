# Reference: https://github.com/qinzheng93/GeoTransformer

import torch.nn as nn
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class LocalPPFTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_heads,
        dropout=None,
        scale_invariant = False,
    ):
        r"""Geometric Transformer (GeoTransformer).
        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            activation_fn: activation function
        """
        super(LocalPPFTransformer, self).__init__()

        self.embedding = PPFStructualEmbedding(hidden_dim, mode = "local", scale_invariant = scale_invariant)
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.transformer = LocalRPEAttentionLayer(d_model=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.residual_path = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        feats,
        node_idx,
        group_idx,
        ppfs
    ):
        r"""Geometric Transformer
        Args:
            feats (Tensor): (N, in_dim)
            node_idx: (M,)
            group_idx: (M, K)
            ppfs (Tensor): (M, K, 4)
        Returns:
            new_feats: torch.Tensor (M, C2)
        """
        pos_embeddings = self.embedding(ppfs) #[M, K, hidden_dims]
        #residual = self.residual_path(feats)
        feats = self.in_proj(feats) #[N, in_dim] -> [N, hidden_dim]
        new_feats, _ = self.transformer(
            feats,
            pos_embeddings,
            node_idx,
            group_idx
        )
        new_feats = self.out_proj(new_feats)
        #new_feats = new_feats + residual
        return new_feats





class LocalRPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(LocalRPEMultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)
        self.proj_vp = nn.Linear(self.d_model, self.d_model)


    def forward(self, input_feats, embed_qk, node_idx, group_idx, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)
        Args:
            input_feats: torch.Tensor (N, C)
            embed_qk: torch.Tensor (N, K, C), relative positional embedding
            node_idx: torch.Tensor (M,), indices of nodes
            group_idx: torch.Tensor(M, K,), indices of groups
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)
        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = self.proj_q(input_feats) # (N, c)
        k = self.proj_k(input_feats) # (N, c)
        v = self.proj_v(input_feats) # (N, c)
        p = self.proj_p(embed_qk) # (M, K, c)
        vp = self.proj_vp(embed_qk)
        #print(q.shape, ' ', node_idx.shape, ' ', node_idx.dtype)

        q = rearrange(q[node_idx], '(b k) (h c) -> b h k c', b=node_idx.shape[0], h=self.num_heads)
        k = rearrange(k[group_idx], 'b k (h c) -> b h k c ', h=self.num_heads)
        v = rearrange(v[group_idx], 'b k (h c) -> b h k c ', h=self.num_heads)
        p = rearrange(p, 'b k (h c) -> b h k c', h=self.num_heads)
        vp = rearrange(vp, 'b k (h c) -> b h k c', h=self.num_heads)

        #q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        #k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        #v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        #p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores_p = torch.einsum('bhnc,bhmc->bhnm', q, p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)

        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)

        hidden_states = torch.matmul(attention_scores, v + vp)

        hidden_states = rearrange(hidden_states, 'b h n c -> (b n) (h c)')
        return hidden_states, attention_scores

class LocalRPEAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(LocalRPEAttentionLayer, self).__init__()
        self.attention = LocalRPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        embed_qk,
        node_idx,
        group_idx,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            embed_qk,
            node_idx,
            group_idx,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
        )
        hidden_states = self.linear(hidden_states)
        output_states = self.norm(hidden_states + input_states[node_idx])
        return output_states, attention_scores



# Reference: https://github.com/qinzheng93/GeoTransformer

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class LocalEmbedding(nn.Module):
    def __init__(self, hidden_dim, scale_invariant = False):
        super(LocalEmbedding, self).__init__()
        self.embed = nn.Linear(4, hidden_dim) #1 + 2 + 2 + 2
        self.scale_invariant = scale_invariant

    def forward(self, ppfs):
        r"""Sinusoidal Positional Embedding.
        Args:
            emb_indices: torch.Tensor (*)
        Returns:
            embeddings: torch.Tensor (*, D)
        """
        d, theta1, theta2, theta3 = ppfs.chunk(4, dim = -1)
        if self.scale_invariant:
            d = d / d.abs().mean(dim=-2, keepdim = True)
        sphere1 = theta1#torch.cat((theta1.cos(), theta1.sin()), dim = -1)
        sphere2 = theta2#torch.cat((theta2.cos(), theta2.sin()), dim = -1)
        sphere3 = theta3#torch.cat((theta3.cos(), theta3.sin()), dim = -1)
        spheres = torch.cat((d,sphere1, sphere2, sphere3), dim = -1) # Is there a better combination here? the distance does not really seem helpful
        embeddings = self.embed(spheres)
        return embeddings


class PPFStructualEmbedding(nn.Module):
    def __init__(self, hidden_dim, mode='local', scale_invariant = False):
        super(PPFStructualEmbedding, self).__init__()
        if mode == 'local':
            self.embedding = LocalEmbedding(hidden_dim, scale_invariant = scale_invariant)
            #self.proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise 'mode should be in [local]'
        self.mode = mode
    def forward(self, ppf):
        embeddings = self.embedding(ppf)
        return embeddings



def calc_ppf_gpu(points, point_normals, patches, patch_normals):
    '''
    Calculate ppf gpu
    points: [n, 3]
    point_normals: [n, 3]
    patches: [n, nsamples, 3]
    patch_normals: [n, nsamples, 3]
    '''
    points = torch.unsqueeze(points, dim=1).expand(-1, patches.shape[1], -1)
    point_normals = torch.unsqueeze(point_normals, dim=1).expand(-1, patches.shape[1], -1)
    vec_d = patches - points #[n, n_samples, 3]
    d = torch.sqrt(torch.sum(vec_d ** 2, dim=-1, keepdim=True)) #[n, n_samples, 1]
    # angle(n1, vec_d)
    y = torch.sum(point_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(point_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle1 = torch.atan2(x, y) / np.pi

    # angle(n2, vec_d)
    y = torch.sum(patch_normals * vec_d, dim=-1, keepdim=True)
    x = torch.cross(patch_normals, vec_d, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle2 = torch.atan2(x, y) / np.pi

    # angle(n1, n2)
    y = torch.sum(point_normals * patch_normals, dim=-1, keepdim=True)
    x = torch.cross(point_normals, patch_normals, dim=-1)
    x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
    angle3 = torch.atan2(x, y) / np.pi

    ppf = torch.cat([d, angle1, angle2, angle3], dim=-1) #[n, samples, 4]
    return ppf


class PPFWrapper(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        dims = [32, 64, 128, 256, conf.descriptor_dim, conf.descriptor_dim, conf.descriptor_dim]
        num_blocks = [1,1,1,1,1,1]
        self.input_ppf_transformer = LocalPPFTransformer(
            input_dim = 1,
            output_dim = dims[0],
            hidden_dim = dims[0],
            scale_invariant = conf.scale_invariant,
            num_heads = 1)
        self.hidden_ppf_transformer = nn.ModuleList([nn.ModuleList([LocalPPFTransformer(
            input_dim = dims[idx],
            output_dim = dims[idx+1],
            hidden_dim = dims[idx],
            scale_invariant = conf.scale_invariant,
            num_heads = 1) for i in range(num_blocks[idx])]) for idx, stride in enumerate(conf.strides)])
        self.strides = conf.strides
        
    def scale_forward_(self, pointcloud, normals, feats = None, index = 0, stride = 1, K = 16):

        #K =  # 33 neighbors were used in RoITr
        pointcloud = pointcloud[:,::stride].contiguous()
        normals = normals[:,::stride].contiguous()
        p_A = pointcloud
        p_S = pointcloud
        B,N,D = p_A.shape
        

        with torch.inference_mode():
            D_A_to_S = torch.cdist(p_A, p_S)
            distances, neighbours = torch.topk(D_A_to_S, k = K, dim = -1, largest = False)
            patches = torch.gather(p_S, dim = 1, index = neighbours[...,None].expand(B,N,K,D).reshape(B,N*K,D)).reshape(B,N,K,D)
            #normals = least_squares_normal_est(patches-p_A[...,None,:],p_A)
            patch_normals = torch.gather(normals, dim = 1, index = neighbours[...,None].expand(B,N,K,D).reshape(B,N*K,D)).reshape(B,N,K,D)
            ppf = calc_ppf_gpu(p_A.reshape(B*N,D), normals.reshape(B*N,D), patches.reshape(B*N,K,D), patch_normals.reshape(B*N,K,D))
        ppf = ppf.clone()
        #idx = torch.arange(B)[:,None].expand(B,N).flatten()
        node_idx = torch.arange(B*N, device = "cuda")
        group_idx = (neighbours.reshape(B,N*K) + N * torch.arange(B, device = "cuda")[:,None]).reshape(B*N, K)

        r"""Geometric Transformer
        Args:
            feats (Tensor): (N, in_dim)
            node_idx: (M,)
            group_idx: (M, K)
            ppfs (Tensor): (M, K, 4)
        Returns:
            new_feats: torch.Tensor (M, C2)
        """
        if feats is None:
            feats = torch.ones_like(p_A[...,:1]).reshape(B*N,1)
            feats = self.input_ppf_transformer(feats, node_idx, group_idx, ppf)
        else:
            feats = feats[:,::stride].reshape(B*N,-1)

        for block in self.hidden_ppf_transformer[index]:
            feats = block(feats, node_idx, group_idx, ppf)
        descriptions = feats.reshape(B,N,-1)
        return pointcloud, normals, descriptions
    
    def internal_forward_(self, pointcloud, normals, feats = None):
        Ks = [16, 32, 32, 32, 64, 64]
        for idx, stride in enumerate(self.strides):
            pointcloud, normals, feats = self.scale_forward_(pointcloud, normals, feats, index = idx, stride = stride, K = Ks[idx])
        B,N,D = pointcloud.shape
        descriptions = feats.reshape(B,N,-1)
        return pointcloud, descriptions
    
    def forward(self, data):
        points_A, points_B = data['pc_A'].contiguous(), data['pc_B'].contiguous()
        normals_A, normals_B = data['normals_A'].contiguous(), data['normals_B'].contiguous()

        keypoints, embeddings = self.internal_forward_(torch.cat((points_A,points_B)), torch.cat((normals_A, normals_B)))

        data['keypoints_A'], data['keypoints_B'] = keypoints.chunk(2)
        data['descriptor_A'], data['descriptor_B'] = embeddings.chunk(2)
        return data

if __name__ == "__main__":
    ppf_model = PPFWrapper().cuda()
    ppf_model(torch.randn(2,1000,3).cuda())
