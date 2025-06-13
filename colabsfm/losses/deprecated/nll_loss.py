import torch
import torch.nn as nn
from colabsfm.metrics import compute_inlier_ratio, compute_mnn_inlier_ratio
from colabsfm.utils import get_correspondences
from colabsfm.models.model import OmniGlue
import colabsfm

class NLLLoss(nn.Module):
    def __init__(self, *args, thr = 0.1, nll_pow = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.thr = thr
        self.tracked_metrics = {}
        self.metric_graph = {}
        self.nll_pow = nll_pow
    def forward(self, data):
        tot_loss = 0
        levels = ["global", "refine"] if "refiner_scores" in data else ["global"]
        for level in levels:
            if level == "global":
                keypoints_A = data["padded_keypoints_A"]
                keypoints_B = data["padded_keypoints_B"]

                tsfm = data["tsfm"].reshape(data["batch_size"], 4, 4)
                mask_A = data["pad_mask_A"]
                mask_B = data["pad_mask_B"]

                
                gt_matches, matchability_A, matchability_B = get_correspondences(keypoints_A, keypoints_B, tsfm, self.thr, mask_A = mask_A, mask_B = mask_B)

                pred_scores = data["scores"]
                pred_matchability = data["matchability"] 
            elif level == "refine":
                keypoints_A = data["node_knn_points_A"]
                keypoints_B = data["node_knn_points_B"]
                mask_A = torch.ones_like(keypoints_A[...,:1]).bool()
                mask_B = torch.ones_like(keypoints_B[...,:1]).bool()
                tsfm_orig = data["tsfm"].reshape(data["batch_size"], 4, 4) #TODO: this is bugged for batch_size > 1
                tsfm = []
                for b in range(data["batch_size"]):
                    num_matches = (data['fine_knn_batch_ids'] == b).sum().item()
                    tsfm.append(tsfm_orig[b:b+1].expand(num_matches,4,4))
                tsfm = torch.cat(tsfm)
                gt_matches, matchability_A, matchability_B = get_correspondences(keypoints_A, keypoints_B, tsfm, 0.5 * self.thr, require_mnn = True)                    
                pred_scores = data["refiner_scores"]
                pred_matchability = data["refiner_matchability"] 
                
            for i in range(len(pred_scores)):
                scores = pred_scores[i]
                pred_matchability_A, pred_matchability_B = pred_matchability[i]
                if pred_matchability_A is not None:
                    mean_bce = (torch.binary_cross_entropy_with_logits(pred_matchability_A, matchability_A.float())[mask_A[...,0]].mean() + 
                            torch.binary_cross_entropy_with_logits(pred_matchability_B, matchability_B.float())[mask_B[...,0]].mean())/2
                else:
                    mean_bce = torch.zeros(1, device = neg_nll.device)

                if len(gt_matches[0]) == 0:
                    tot_loss = tot_loss + mean_bce
                    continue
                else:
                    neg_nll = -scores[gt_matches[0],gt_matches[1],gt_matches[2]]     
                mean_neg_nll = neg_nll.mean().item()
                inlier_nll = (neg_nll < 3).float().mean()
                self.tracked_metrics[f"{level}_mean_neg_nll_{i}"] = (0.9 * self.tracked_metrics.get(f"{level}_mean_neg_nll_{i}", mean_neg_nll) + 0.1 * mean_neg_nll)
                self.tracked_metrics[f"{level}_inlier_nll_{i}"] = (0.9 * self.tracked_metrics.get(f"{level}_inlier_nll_{i}", inlier_nll) + 0.1 * inlier_nll)
                self.tracked_metrics[f"{level}_mean_bce_{i}"] = (0.9 * self.tracked_metrics.get(f"{level}_mean_bce_{i}", mean_bce.item()) + 0.1 * mean_bce.item())

                tot_loss = tot_loss + (neg_nll**self.nll_pow).mean() + mean_bce
            for b in range(data["batch_size"]):
                if level == "global":
                    p, q = data['coarse_src_corr_pts'][data['coarse_corr_batch_ids'] == b], data['coarse_tgt_corr_pts'][data['coarse_corr_batch_ids'] == b]
                elif level == "local":
                    p, q = data['src_corr_pts'][data['corr_batch_ids'] == b], data['tgt_corr_pts'][data['corr_batch_ids'] == b]
                IR = compute_inlier_ratio(p, q, data["tsfm"].reshape(data["batch_size"], 4, 4)[b])
                #mnn_IR = compute_mnn_inlier_ratio(p, q, data["tsfm"].reshape(data["batch_size"], 4, 4)[b])
            self.tracked_metrics[f"{level}_IR"] = (0.9 * self.tracked_metrics.get(f"{level}_IR", IR) + 0.1 * IR)
            #self.tracked_metrics[f"{level}_mnn_IR_{i}"] = (0.9 * self.tracked_metrics.get(f"{level}_mnn_IR_{i}", mnn_IR) + 0.1 * mnn_IR)
        if colabsfm.LOGGER is not None and (colabsfm.GLOBAL_STEP % 10 == 0):
            colabsfm.LOGGER.add_scalars('data/train_loss', {"global_nll": self.tracked_metrics[f"global_mean_neg_nll_{i}"], 
                                                           "global_bce": self.tracked_metrics[f"global_mean_bce_{i}"]}, global_step = colabsfm.GLOBAL_STEP)
            colabsfm.LOGGER.add_scalars('data/train_loss', {"refine_nll": self.tracked_metrics[f"refine_mean_neg_nll_{i}"], 
                                                           "refine_bce": self.tracked_metrics[f"refine_mean_bce_{i}"]}, global_step = colabsfm.GLOBAL_STEP)
            colabsfm.LOGGER.add_scalars('data/train_metrics', {"refine_IR": self.tracked_metrics.get(f"refine_IR", 0),
                                                              "global_IR": self.tracked_metrics[f"global_IR"]}, global_step = colabsfm.GLOBAL_STEP)
        return tot_loss

