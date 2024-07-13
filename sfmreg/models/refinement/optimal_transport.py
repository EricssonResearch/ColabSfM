import torch
from torch import nn
from typing import Optional
import numpy as np


class LearnableLogOptimalTransport(nn.Module):
    '''
    Optimal Transport Layer with learnable thresholds
    Reference: GeoTransformer, Zheng Qin et al.
    '''
    def __init__(self, conf, inf=1e6):
        super(LearnableLogOptimalTransport, self).__init__()
        self.num_iter = conf.num_iter
        self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.)))
        self.inf = inf

    def log_sinkhorn_normalization(self, scores, log_mu, log_nu):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(self.num_iter):
            u = log_mu - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
        return scores + u.unsqueeze(2) + v.unsqueeze(1)

    def forward(self, data):
        r"""
        Optimal transport with Sinkhorn.
        :param scores: torch.Tensor (B, M, N)
        :param row_masks: torch.Tensor (B, M)
        :param col_masks: torch.Tensor (B, N)
        :return matching_scores: torch.Tensor (B, M+1, N+1)
        """

        desc0, desc1 = data['node_knn_feats_A'], data['node_knn_feats_B']
        mask0, mask1 = None, None
        matching_scores = torch.einsum('bnd,bmd->bnm', desc0,
                                       desc1)  # (P, K, K)
        scores = matching_scores / desc0.shape[-1] ** 0.5
        row_masks = torch.ones_like(matching_scores[:,:,0]).bool()
        col_masks = torch.ones_like(matching_scores[:,0,:]).bool()

        # outputs: scores, row_masks, col_masks
        batch_size, num_row, num_col = scores.shape
        ninf = torch.tensor(-self.inf).cuda()

        padded_row_masks = torch.zeros(batch_size, num_row + 1, dtype=torch.bool).cuda()
        padded_row_masks[:, :num_row] = ~row_masks
        padded_col_masks = torch.zeros(batch_size, num_col + 1, dtype=torch.bool).cuda()
        padded_col_masks[:, :num_col] = ~col_masks

        padded_col = self.alpha.expand(batch_size, num_row, 1)
        padded_row = self.alpha.expand(batch_size, 1, num_col + 1)
        padded_scores = torch.cat([torch.cat([scores, padded_col], dim=-1), padded_row], dim=1)

        padded_score_masks = padded_row_masks.unsqueeze(2) +  padded_col_masks.unsqueeze(1)
        padded_scores[padded_score_masks] = ninf

        num_valid_row = row_masks.float().sum(1)
        num_valid_col = col_masks.float().sum(1)
        norm = -torch.log(num_valid_row + num_valid_col)  # (B,)

        log_mu = torch.empty(batch_size, num_row + 1).cuda()
        log_mu[:, :num_row] = norm.unsqueeze(1)
        log_mu[:, num_row] = torch.log(num_valid_col) + norm
        log_mu[padded_row_masks] = ninf

        log_nu = torch.empty(batch_size, num_col + 1).cuda()
        log_nu[:, :num_col] = norm.unsqueeze(1)
        log_nu[:, num_col] = torch.log(num_valid_row) + norm
        log_nu[padded_col_masks] = ninf

        ot_scores = self.log_sinkhorn_normalization(padded_scores, log_mu, log_nu)
        ot_scores = ot_scores - norm.unsqueeze(1).unsqueeze(2)
        data['refiner_scores'] = [ot_scores[:,:-1, :-1]]
        data['refiner_matchability'] = [(None, None)]
        return data

        return outputs

    def __repr__(self):
        format_string = self.__class__.__name__ + '(num_iter={})'.format(self.num_iter)
        return format_string