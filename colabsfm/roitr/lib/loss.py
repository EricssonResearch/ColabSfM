# Reference: https://github.com/qinzheng93/GeoTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..lib.utils import square_distance
import colabsfm
from ..lib.utils import to_o3d_pcd
from ..registration.benchmark_utils import ransac_pose_estimation_correspondences
from ..registration.benchmark import rotation_error, translation_error
import numpy as np
import math
from scipy.spatial import cKDTree
# from colabsfm.roitr.lib.utils import get_correspondences

def apply_transform(points: np.ndarray, transform: np.ndarray):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    
    return points

def get_correspondences(ref_points, src_points, transform, matching_radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    corr_indices = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices])
    return corr_indices

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
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss_positive_margin,
            cfg.coarse_loss_negative_margin,
            cfg.coarse_loss_positive_optimal,
            cfg.coarse_loss_negative_optimal,
            cfg.coarse_loss_log_scale,
        )
        self.positive_overlap = cfg.coarse_loss_positive_overlap
        self.cfg = cfg

    def forward(self, output_dict):
        tgt_feats = output_dict['tgt_node_feats']
        src_feats = output_dict['src_node_feats']

        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']


        gt_tgt_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(square_distance(tgt_feats[None, ::], src_feats[None, ::])[0])

        #print(tgt_feats)
        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_tgt_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps

        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)
        if colabsfm.LOGGER is not None:
            colabsfm.LOGGER.add_scalars('data/roitr_train_loss', {"circle_loss": loss.item()}, global_step = colabsfm.GLOBAL_STEP)
        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss_positive_radius
        self.mnn = cfg.fine_loss_use_mnn

    def forward(self, output_dict, data_dict):
        tgt_node_corr_knn_points = output_dict['tgt_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        tgt_node_corr_knn_masks = output_dict['tgt_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        rot = data_dict['rot'][0]
        trans = data_dict['trans'][0]
        #src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)

        src_node_corr_knn_points = torch.matmul(src_node_corr_knn_points, rot.T) + (trans.T)[None, ::]
        dists = square_distance(tgt_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = tgt_node_corr_knn_masks.unsqueeze(2) * src_node_corr_knn_masks.unsqueeze(1)
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = gt_corr_map * gt_masks
        if self.mnn:
            gt_corr_map = gt_corr_map * \
                (dists < dists.topk(k = 3, largest = False, dim=1).values[:,2:]) * \
                (dists < dists.topk(k = 3, largest = False, dim=2).values[:,:,2:])
        slack_row_labels = torch.eq(gt_corr_map.sum(2), 0) * tgt_node_corr_knn_masks
        slack_col_labels = torch.eq(gt_corr_map.sum(1), 0) * src_node_corr_knn_masks
        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()
        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.coarse_loss_weight
        self.weight_fine_loss = cfg.fine_loss_weight
        self.weight_occ_loss = cfg.occ_loss_weight

    def forward(self, output_dict, data_dict):
        if 'src_node_feats' in output_dict:
            coarse_loss = self.coarse_loss(output_dict)
            fine_loss = self.fine_loss(output_dict, data_dict)

            loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss
            if colabsfm.LOGGER is not None:
                colabsfm.LOGGER.add_scalars('data/roitr_train_loss', {"fine_loss": fine_loss.item()}, global_step = colabsfm.GLOBAL_STEP)
            colabsfm.GLOBAL_STEP = colabsfm.GLOBAL_STEP + 1
        else:
            coarse_loss, fine_loss, loss = torch.zeros([1]), torch.zeros([1]), torch.zeros([1])
        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
            'o_loss': 0. * fine_loss
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval_acceptance_overlap
        self.acceptance_radius = cfg.eval_acceptance_radius

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        tgt_length_c = output_dict['tgt_nodes'].shape[0]
        src_length_c = output_dict['src_nodes'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_tgt_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(tgt_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_tgt_node_corr_indices, gt_src_node_corr_indices] = 1.0

        tgt_node_corr_indices = output_dict['tgt_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[tgt_node_corr_indices, src_node_corr_indices].mean()
        if torch.isnan(precision):
            print("hej")
        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        rot, trans = data_dict['rot'][0], data_dict['trans'][0]
        tgt_corr_points = output_dict['tgt_corr_points']
        src_corr_points = output_dict['src_corr_points']
        if src_corr_points.shape[0] == 0:
            precision = 0.
        else:
            src_corr_points = torch.matmul(src_corr_points, rot.T) + trans.T
            corr_distances = torch.linalg.norm(tgt_corr_points - src_corr_points, dim=1)
            precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    def forward(self, output_dict, data_dict):
        if 'src_node_feats' in output_dict:
            c_precision = self.evaluate_coarse(output_dict)
        else:
            c_precision = 0.
        f_precision = self.evaluate_fine(output_dict, data_dict)
        return {
            'PIR': c_precision,
            'IR': f_precision,
            'FMR': f_precision > 0.05,
            'num_matches': output_dict['src_corr_points'].shape[0],
        }

class EvaluatorRegistration(nn.Module):
    def __init__(self, cfg,return_pose=False):
        super(EvaluatorRegistration, self).__init__()
        self.acceptance_overlap = cfg.eval_acceptance_overlap
        self.acceptance_radius = cfg.eval_acceptance_radius
        self.n_points = cfg.n_points_registration
        self.distance_threshold = cfg.distance_threshold
        self.ransac_n = cfg.ransac_n
        self.rot_thresh = cfg.rot_thresh
        self.trans_thresh = cfg.trans_thresh
        self.mode = cfg.colabsfm_mode
        self.return_pose = return_pose

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        tgt_length_c = output_dict['tgt_nodes'].shape[0]
        src_length_c = output_dict['src_nodes'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_tgt_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(tgt_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_tgt_node_corr_indices, gt_src_node_corr_indices] = 1.0

        tgt_node_corr_indices = output_dict['tgt_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[tgt_node_corr_indices, src_node_corr_indices].mean()
        if torch.isnan(precision):
            print("hej")
        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        rot, trans = data_dict['rot'][0], data_dict['trans'][0]
        tgt_corr_points = output_dict['tgt_corr_points']
        src_corr_points = output_dict['src_corr_points']
        if src_corr_points.shape[0] == 0:
            precision = 0.
            gt_corrs = []
        else:
            src_corr_points = torch.matmul(src_corr_points, rot.T) + trans.T
            corr_distances = torch.linalg.norm(tgt_corr_points - src_corr_points, dim=1)
            precision =  torch.lt(corr_distances, self.acceptance_radius).float().mean()
            gt_corrs = (corr_distances.cpu().numpy() < self.acceptance_radius)
        return precision, gt_corrs
    
    @torch.no_grad()
    def umeyama(self,ptsA, ptsB):
        # assume array shape is Nx3
        n = ptsA.shape[0]

        # compute centroids
        centerA = np.mean(ptsA,axis=0)
        centerB = np.mean(ptsB,axis=0)

        # remove center point
        pt3DCenteredA = ptsA - centerA.reshape(1,-1)
        pt3DCenteredB = ptsB - centerB.reshape(1,-1)

        # compute covariance matrix
        Sxy =  (pt3DCenteredB.T @ pt3DCenteredA)/n

        # get rotation
        S = np.eye(3)
        if np.linalg.det(Sxy) < 0:
            S[2,2] = -1
        U,D,Vh = np.linalg.svd(Sxy)
        R = U @ S @ Vh
        
        # get scale
        sigmaX = np.mean(np.sum(pt3DCenteredA * pt3DCenteredA, axis=1))
        scale = np.trace(np.diag(D) @ S)/sigmaX
        
        # get translation
        translation = centerB - scale*(R@centerA)

        return R, translation, scale
    
    @torch.no_grad()
    def ransac(self,ptsA, ptsB,npoints,threshold,max_iter=10000,compute_rmse=False):
        R_sol = np.eye(3)
        t_sol = np.zeros(3)
        s_sol = 0
        num_inliers = 0
        inlier_ratio = 0
        maxIter = max_iter
        iter = 0
        if compute_rmse:
            inlierRMSE = np.inf
            rmse = np.inf
        while iter < maxIter:
            # select randomly 3 points
            sel_idx = np.random.choice(ptsB.shape[0], npoints, replace=False)
            #compute model
            rot, trans, scale = self.umeyama(ptsA[sel_idx,:],ptsB[sel_idx])
            # eval model
            ptsAB = scale*(ptsA @ rot.T) + trans
            dists = np.linalg.norm(ptsAB - ptsB, axis=1)
            inlier_mask = (dists < threshold)
            nin = inlier_mask.sum()
            if nin > num_inliers:
                if compute_rmse:
                    inlierRMSE = np.sqrt(np.mean(dists[inlier_mask]**2))
                    rmse = np.sqrt(np.mean(dists**2))
                num_inliers = nin
                inlier_ratio = num_inliers/ptsB.shape[0]
                R_sol = rot
                t_sol = trans
                s_sol = scale
                maxIter = math.ceil(min(max_iter,np.log(1-.99)/np.log(1-inlier_ratio**npoints)))
            iter += 1

        if compute_rmse:
            return R_sol, t_sol, s_sol, inlierRMSE, rmse
        else:
            return R_sol, t_sol, s_sol

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict, gt_corrs):
        # get registration results
        rot_gt, trans_gt = data_dict['rot'][0], data_dict['trans'][0]
        tgt_corr_points = output_dict['tgt_corr_points']
        src_corr_points = output_dict['src_corr_points']
        if output_dict['src_corr_points'].shape[0] > self.n_points:
            prob = output_dict['corr_scores'] / torch.sum(output_dict['corr_scores'])
            # sel_idx = np.random.choice(output_dict['src_corr_points'].shape[0], self.n_points, replace=False, p=prob.cpu().numpy())
            sel_idx = np.random.choice(output_dict['src_corr_points'].shape[0], self.n_points, replace=False)
            src_corr_points, tgt_corr_points = src_corr_points[sel_idx], tgt_corr_points[sel_idx]
        correspondences = torch.from_numpy(np.arange(src_corr_points.shape[0])[:, np.newaxis]).expand(-1, 2)
        if self.mode == "sim3":
            # rot_est, t_est, scale_est, inlier_rmse, rmse = self.ransac(src_corr_points.cpu().numpy(), tgt_corr_points.cpu().numpy(),self.ransac_n,self.distance_threshold)
            rot_est, t_est, scale_est = self.ransac(src_corr_points.cpu().numpy(), tgt_corr_points.cpu().numpy(),self.ransac_n,self.distance_threshold,compute_rmse=False)
            # gt
            U, S, Vh = np.linalg.svd(rot_gt.cpu().numpy(), full_matrices=True)
            scale_gt = S[0]
            rot_gt = torch.from_numpy(U @ Vh).to(trans_gt.get_device()).float()
            if torch.linalg.det(rot_gt) < 0:
                rot_gt *= -1.
            # set translation
            # t_est = Test[:3,3]
            # scale error
            scale_error = np.abs(scale_est-scale_gt)
        else:
            # Test, inlier_rmse = ransac_pose_estimation_correspondences(src_corr_points, tgt_corr_points, correspondences, distance_threshold=self.distance_threshold,ransac_n=self.ransac_n)
            Test = ransac_pose_estimation_correspondences(src_corr_points, tgt_corr_points, correspondences, distance_threshold=self.distance_threshold,ransac_n=self.ransac_n)
            rot_est = Test[:3,:3]
            t_est = Test[:3,3]
            # rot_est, t_est, scale_est = self.ransac(src_corr_points.cpu().numpy(), tgt_corr_points.cpu().numpy(),self.ransac_n,self.distance_threshold)
            scale_error = 0
            scale_est = 1
        
        src_pts = output_dict['tgt_corr_points'].cpu().numpy()
        tgt_pts = output_dict['src_corr_points'].cpu().numpy()
        ptsAB = scale_est*(src_pts @ rot_est.T) + t_est
        dists = np.linalg.norm(ptsAB[gt_corrs,:] - tgt_pts[gt_corrs,:], axis=1)
        rmse = np.sqrt(np.mean(dists**2))
        # src_pcd, tgt_pcd = data_dict['src_points'].contiguous().cpu().numpy(), data_dict['tgt_points'].contiguous().cpu().numpy()
        # Tgt = np.eye(4)
        # Tgt[:3,:3] = rot_gt.cpu().numpy()
        # Tgt[:3,3] = trans_gt.cpu().numpy()
        # correspondences = get_correspondences(src_pcd,tgt_pcd,Test,0.1)
        # ptsAB = scale_est*(src_pcd[correspondences[:,0]] @ rot_est.T) + t_est
        # dists = np.linalg.norm(ptsAB - tgt_pcd[correspondences[:,1]], axis=1)
        # rmse = np.sqrt(np.mean(dists**2))
            
        rot_est = torch.from_numpy(rot_est).to(trans_gt.get_device()).float()
        t_est = torch.from_numpy(t_est).to(trans_gt.get_device())
        rot_error = rotation_error(rot_est.unsqueeze(0),rot_gt.unsqueeze(0))
        trans_error = translation_error(t_est[None,:,None],trans_gt[None,:,None])
        
        return rot_error, trans_error, scale_error, rmse, rot_est, t_est

    def forward(self, output_dict, data_dict):
        if 'src_node_feats' in output_dict:
            c_precision = self.evaluate_coarse(output_dict)
        else:
            c_precision = 0.
        f_precision, gt_corrs = self.evaluate_fine(output_dict, data_dict)
        
        rot_error, trans_error, scale_error, rmse, rot_est, t_est = self.evaluate_registration(output_dict, data_dict, gt_corrs)
        
        if self.return_pose:
            return {
                'PIR': c_precision,
                'IR': f_precision,
                'FMR': f_precision > 0.05,
                'num_matches': output_dict['src_corr_points'].shape[0],
                'rot_error': rot_error.squeeze(),
                'trans_error': trans_error.squeeze(),
                'RR': rot_error.squeeze() < self.rot_thresh and trans_error.squeeze() < self.trans_thresh and scale_error < 0.05,
                # 'scale_error': scale_error,
                # 'inlier_rmse': inlier_rmse,
                # 'RR_in_rmse': inlier_rmse < self.distance_threshold**2,
                'rmse': rmse,
                'RR_rmse': rmse < 0.2,
                'rot': rot_est,
                'trans': t_est
            }
        else:
            return {
                'PIR': c_precision,
                'IR': f_precision,
                'FMR': f_precision > 0.05,
                'num_matches': output_dict['src_corr_points'].shape[0],
                'rot_error': rot_error.squeeze(),
                'trans_error': trans_error.squeeze(),
                'RR': rot_error.squeeze() < self.rot_thresh and trans_error.squeeze() < self.trans_thresh and scale_error < 0.05,
                # 'scale_error': scale_error,
                # 'inlier_rmse': inlier_rmse,
                # 'RR_in_rmse': inlier_rmse < self.distance_threshold**2,
                'rmse': rmse,
                'RR_rmse': rmse < 0.2
            }