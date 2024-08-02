# Reference: https://github.com/qinzheng93/GeoTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

from colabsfm.utils import get_correspondences

def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def weighted_circle_loss(
    pos_masks,
    neg_masks,
    feat_dists,
    pos_margin,
    neg_margin,
    pos_optimal,
    neg_optimal,
    log_scale,
    pos_scales=None,
    neg_scales=None,
):
    # get anchors that have both positive and negative pairs
    row_masks = (torch.gt(pos_masks.sum(-1), 0) & torch.gt(neg_masks.sum(-1), 0)).detach()
    col_masks = (torch.gt(pos_masks.sum(-2), 0) & torch.gt(neg_masks.sum(-2), 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weights = feat_dists - 1e5 * (~pos_masks).float()  # mask the non-positive
    pos_weights = pos_weights - pos_optimal  # mask the uninformative positive
    pos_weights = torch.maximum(torch.zeros_like(pos_weights), pos_weights)
    if pos_scales is not None:
        pos_weights = pos_weights * pos_scales
    pos_weights = pos_weights.detach()

    neg_weights = feat_dists + 1e5 * (~neg_masks).float()  # mask the non-negative
    neg_weights = neg_optimal - neg_weights  # mask the uninformative negative
    neg_weights = torch.maximum(torch.zeros_like(neg_weights), neg_weights)
    if neg_scales is not None:
        neg_weights = neg_weights * neg_scales
    neg_weights = neg_weights.detach()

    loss_pos_row = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-1)
    loss_pos_col = torch.logsumexp(log_scale * (feat_dists - pos_margin) * pos_weights, dim=-2)

    loss_neg_row = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-1)
    loss_neg_col = torch.logsumexp(log_scale * (neg_margin - feat_dists) * neg_weights, dim=-2)

    loss_row = F.softplus(loss_pos_row + loss_neg_row) / log_scale
    loss_col = F.softplus(loss_pos_col + loss_neg_col) / log_scale
    loss = (loss_row[row_masks].mean() + loss_col[col_masks].mean()) / 2

    return loss


class WeightedCircleLoss(nn.Module):
    def __init__(self, pos_margin, neg_margin, pos_optimal, neg_optimal, log_scale):
        super(WeightedCircleLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal
        self.log_scale = log_scale

    def forward(self, pos_masks, neg_masks, feat_dists, pos_scales=None, neg_scales=None):
        return weighted_circle_loss(
            pos_masks,
            neg_masks,
            feat_dists,
            self.pos_margin,
            self.neg_margin,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
            pos_scales=pos_scales,
            neg_scales=neg_scales,
        )


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg, thr = 0.1):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss_positive_margin,
            cfg.coarse_loss_negative_margin,
            cfg.coarse_loss_positive_optimal,
            cfg.coarse_loss_negative_optimal,
            cfg.coarse_loss_log_scale,
        )
        self.positive_overlap = cfg.coarse_loss_positive_overlap
        self.thr = thr

    def forward(self, output_dict):
        output_dict["descriptor_A"] = F.normalize(output_dict["descriptor_A"], p=2, dim=-1)
        output_dict["descriptor_B"] = F.normalize(output_dict["descriptor_B"], p=2, dim=-1)
        src_feats = output_dict['descriptor_A']
        tgt_feats = output_dict['descriptor_B']
        gt_matches, matchability_A, matchability_B = get_correspondences(output_dict['keypoints_A'], output_dict['keypoints_B'], output_dict['tsfm'], self.thr)
        #gt_node_corr_indices = output_dict['gt_node_corr_indices']
        #gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']


        #gt_tgt_node_corr_indices = gt_node_corr_indices[:, 0]
        #gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(square_distance(tgt_feats, src_feats)[0])

        #print(tgt_feats)
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_matches[2], gt_matches[1]] = 1

        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss