import torch
import torch.nn as nn
import torch.nn.functional as F
import colabsfm
import sys

sys.path.append("colabsfm/OverlapPredator")
from datasets.dataloader import collate_fn_descriptor
from lib.benchmark_utils import mutual_selection, get_correspondences, to_o3d_pcd, to_tsfm
from lib.loss import MetricLoss

class MetricLossWraper(nn.Module):
    def __init__(self, cfg):
        super(MetricLossWraper, self).__init__()
        self.metric_loss = MetricLoss(cfg)
        self.w_circle_loss = cfg.w_circle_loss
        self.w_overlap_loss = cfg.w_overlap_loss
        self.w_saliency_loss = cfg.w_saliency_loss 
        # self.desc_loss = cfg.desc_loss
        self.overlap_radius = cfg.overlap_radius
        
    def forward(self, output_dict, data_dict):
        src_pcd = data_dict['src_points'].contiguous()
        tgt_pcd = data_dict['tgt_points'].contiguous()
        rot, trans = data_dict['rot'][0], data_dict['trans'][0]
        tsfm = to_tsfm(rot.cpu(), trans.cpu())
        correspondence = get_correspondences(to_o3d_pcd(src_pcd.cpu().numpy()), to_o3d_pcd(tgt_pcd.cpu().numpy()), tsfm,self.overlap_radius)
        tsfm = torch.from_numpy(tsfm).to("cuda:0")
        correspondence = correspondence.to("cuda:0")
        
        ###################################################
        # get loss
        stats = self.metric_loss(src_pcd, tgt_pcd, output_dict['src_point_feats'], output_dict['tgt_point_feats'],correspondence, rot, torch.unsqueeze(trans, 1), output_dict['scores_overlap'], output_dict['scores_saliency'])

        loss = stats['circle_loss'] * self.w_circle_loss + stats['overlap_loss'] * self.w_overlap_loss + stats['saliency_loss'] * self.w_saliency_loss
        
        if colabsfm.LOGGER is not None:
            colabsfm.LOGGER.add_scalars('data/predator_train_loss', {"loss": loss.item()}, global_step = colabsfm.GLOBAL_STEP)
        colabsfm.GLOBAL_STEP = colabsfm.GLOBAL_STEP + 1
        
        return {
            'loss': loss,
            'c_loss': stats['circle_loss'],
            'o_loss': stats['overlap_loss'],
            'f_loss': stats['saliency_loss']
        }
