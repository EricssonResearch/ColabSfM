from typing import Any, Mapping
import warnings, sys
import numpy as np
import torch, time, gc, tqdm
from torch import nn
import open3d as o3d
import math
from easydict import EasyDict as edict

from omnireg.utils import get_correspondences, estimate_normals, transform
from omnireg.benchmarks.benchmark_utils import to_array, to_o3d_pcd
from omnireg.geotransformer.utils.torch import to_cuda, release_cuda

from omnireg.geotransformer.modules.ops import point_to_node_partition, index_select
from omnireg.geotransformer.modules.registration import get_node_correspondences

sys.path.append("omnireg/OverlapPredator")
from datasets.dataloader import collate_fn_descriptor
from lib.benchmark_utils import mutual_selection, get_correspondences

class PreadtorWrap(nn.Module):
    def __init__(self, model: nn.Module, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.config = config
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict, assign)

    @torch.inference_mode()
    def register(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None, viewpoints_A = None, viewpoints_B = None, shared_scale = False):
        pointcloud_A, pointcloud_B = pointcloud_A.float(), pointcloud_B.float()
        if len(pointcloud_A) > 30_000:
            warnings.warn("Pointcloud A larger than 30_000 samples, downsampling to 30_000")
            inds_A = np.random.choice(len(pointcloud_A), size = 30_000, replace = False)
            pointcloud_A = pointcloud_A[inds_A]
            viewpoints_A = viewpoints_A[inds_A] if viewpoints_A is not None else None
            normals_A = normals_A[inds_A] if normals_A is not None else None
        if len(pointcloud_B) > 30_000:
            warnings.warn("Pointcloud B larger than 30_000 samples, downsampling to 30_000")
            inds_B = np.random.choice(len(pointcloud_B), size = 30_000, replace = False)
            pointcloud_B = pointcloud_B[inds_B]
            viewpoints_B = viewpoints_B[inds_B] if viewpoints_B is not None else None
            normals_B = normals_B[inds_B] if normals_B is not None else None


        self.train(False)
        results = self.match(pointcloud_A, pointcloud_B, 
                             normals_A = normals_A, normals_B = normals_B, 
                             viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B, 
                             shared_scale = shared_scale)
        print("errors scaled, no transform", ((results['src_corr_points'] - results['tgt_corr_points']).norm(dim=-1) < 0.1).float().mean())
        print(len(results['src_corr_points']), "num corrs")
        results = self.solve(results, transform_mode = "se3")
        return results

    def estimate_scale(self, pointcloud):
        pointcloud_mean = pointcloud.mean(dim=0,keepdim=True)
        corrs = (pointcloud-pointcloud_mean).T @ (pointcloud-pointcloud_mean) / len(pointcloud)
        biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().float()
        scale = biggest_singular / np.sqrt(2)
        return scale



    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals, rot, trans, src_raw_pcd):
        batched_input = [[
                        src_pcd.cpu().float(),                      #src_pcd
                        tgt_pcd.cpu().float(),                      #tgt_pcd
                        src_feats.cpu().float(),                    #src_feats
                        tgt_feats.cpu().float(),                    #tgt_feats
                        rot.cpu().numpy().astype(np.float32),       #rot
                        trans.cpu().numpy().astype(np.float32),     #trans
                        torch.ones(1).long(),                       #matching_inds,
                        src_pcd.cpu().numpy().astype(np.float32),   #src_pcd_raw
                        tgt_pcd.cpu().numpy().astype(np.float32),   #tgt_pcd_raw
                        torch.ones(1)                               #sample
                        ]]
        
        # print("number of features", src_feats.shape)
        data_dict = collate_fn_descriptor(batched_input, self.config, self.config.neighborhood_limits)
        # print("points after collate", data_dict['stack_lengths'][-1][0])
        if data_dict['stack_lengths'][-1][0] > 900:
            # self.config.first_subsampling_dl = .05
            config_down = edict(self.config)
            config_down.first_subsampling_dl = .05
            neighborhood_limits = [None]*len(self.config.neighborhood_limits)
            for iter,nl in enumerate(self.config.neighborhood_limits):
                neighborhood_limits[iter] = math.ceil(nl*.8)
            # print(neighborhood_limits)
            
            data_dict = collate_fn_descriptor(batched_input, config_down, neighborhood_limits)
            # print("points after collate", data_dict['stack_lengths'][-1][0])
            
        if data_dict['stack_lengths'][-1][0] < 50 or data_dict['stack_lengths'][-1][0] > 1500:
            return {}

        # print("neighbors",self.config.neighborhood_limits, self.config.first_subsampling_dl)
        
        # This IS fast and optimal
        data_dict = to_cuda(data_dict)
        feats, scores_overlap, scores_saliency = self.model.forward(data_dict)

        len_src = src_pcd.shape[0]
        src_feats, tgt_feats = feats[:len_src], feats[len_src:]
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()
        
        ########################################
        # do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency

        if(src_pcd.size(0) > self.config.n_points):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= self.config.n_points, replace=False, p=probs)
            src_node_pcd, src_node_feats = src_pcd[idx], src_feats[idx]
        if(tgt_pcd.size(0) > self.config.n_points):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= self.config.n_points, replace=False, p=probs)
            tgt_node_pcd, tgt_node_feats = tgt_pcd[idx], tgt_feats[idx]
        
        scores = torch.matmul(src_node_feats.to(self.config.device), tgt_node_feats.transpose(0,1).to(self.config.device)).cpu()
        selection = mutual_selection(scores[None,:,:])[0]
        row_sel, col_sel = np.where(selection)
        src_corr_points, tgt_corr_points = src_node_pcd[row_sel], tgt_node_pcd[col_sel]

        transf = np.eye(4)
        transf[:3, :3] = rot.cpu().numpy()
        transf[:3, 3] = trans.cpu().numpy()

        # calculate overlap
        # _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
        #     tgt_pcd, tgt_corr_points, min(64, tgt_pcd.shape[0])
        # )
        # _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
        #     src_pcd, src_corr_points, min(64, src_pcd.shape[0])
        # )

        # ref_padded_points_f = torch.cat([tgt_pcd, torch.zeros_like(tgt_pcd[:1])], dim=0)
        # src_padded_points_f = torch.cat([src_pcd, torch.zeros_like(src_pcd[:1])], dim=0)
        # ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        # src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        # gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
        #     tgt_corr_points,
        #     src_corr_points,
        #     ref_node_knn_points,
        #     src_node_knn_points,
        #     torch.from_numpy(transf).float().to(self.config.device),
        #     0.05,
        #     ref_masks=ref_node_masks,
        #     src_masks=src_node_masks,
        #     ref_knn_masks=ref_node_knn_masks,
        #     src_knn_masks=src_node_knn_masks,
        # )


        outputs = {
            "src_points": src_pcd,                 #torch.Size([N, 3])
            "tgt_points": tgt_pcd,                 #torch.Size([M, 3])
            "src_nodes": src_node_pcd,                  #torch.Size([n, 3])
            "tgt_nodes": tgt_node_pcd,                  #torch.Size([m, 3])
            "src_point_feats": src_feats,            #torch.Size([N, 64])
            "tgt_point_feats": tgt_feats,            #torch.Size([M, 64])
            # "src_node_feats": src_node_feats,             #torch.Size([n, 256])
            # "tgt_node_feats": tgt_node_feats,             #torch.Size([m, 256])
            # "gt_node_corr_indices": gt_node_corr_indices, #torch.Size([GT, 2]) ??????????????????
            # "gt_node_corr_overlaps": gt_node_corr_overlaps, #torch.Size([GT])    ??????????????????
            # "src_node_corr_indices": row_sel,      #torch.Size([256])
            # "tgt_node_corr_indices": col_sel,      #torch.Size([256])
            "tgt_corr_points": tgt_corr_points,            #torch.Size([1838, 3])
            "src_corr_points": src_corr_points,            #torch.Size([1838, 3])
            "scores_overlap": scores_overlap, 
            "scores_saliency": scores_saliency
        }
        
        return outputs

    @torch.inference_mode()
    def match(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None, 
              viewpoints_A = None, viewpoints_B = None, features_A = None, features_B = None, 
              shared_scale = False, voxel_downsample = False, voxel_size = 0.02, device = "cuda"):
        self.train(False)
        assert normals_A is not None or viewpoints_A is not None, "Please provide either the viewpoint or a precomputed normal"
        assert normals_B is not None or viewpoints_B is not None, "Please provide either the viewpoint or a precomputed normal"

        scale_A = self.estimate_scale(pointcloud_A).item()
        pointcloud_A = pointcloud_A / scale_A
        viewpoints_A = viewpoints_A / scale_A  
        if shared_scale:
            scale_B = scale_A
        else:
            scale_B = self.estimate_scale(pointcloud_B).item()
        pointcloud_B = pointcloud_B / scale_B
        viewpoints_B = viewpoints_B / scale_B
        
        if voxel_downsample:
            pcd_A = to_o3d_pcd(pointcloud_A.cpu())
            pcd_A, _, inds_A = pcd_A.voxel_down_sample_and_trace(voxel_size, pcd_A.get_min_bound(), pcd_A.get_max_bound())
            inds_A = np.array([i[0] for i in inds_A])# take the first ind only,
            if normals_A is not None:
                normals_A = normals_A[inds_A]
            else:
                viewpoints_A = viewpoints_A[inds_A]
            pointcloud_A = torch.from_numpy(np.array(pcd_A.points)).float().to(device)

            pcd_B = to_o3d_pcd(pointcloud_B.cpu())
            pcd_B, _, inds_B = pcd_B.voxel_down_sample_and_trace(voxel_size, pcd_B.get_min_bound(), pcd_B.get_max_bound())
            inds_B = np.array([i[0] for i in inds_B])# take the first ind only,
            if normals_B is not None:
                normals_B = normals_B[inds_B]
            else:
                viewpoints_B = viewpoints_B[inds_B]
            pointcloud_B = torch.from_numpy(np.array(pcd_B.points)).float().to(device)

        if features_A is None:
            features_A, features_B = torch.ones_like(pointcloud_A[...,:1]), torch.ones_like(pointcloud_B[...,:1])

        if normals_A is None or normals_B is None:
            normals_A = estimate_normals(pointcloud_A, view_point=viewpoints_A).float()
            normals_B = estimate_normals(pointcloud_B, view_point=viewpoints_B).float()
        results = self.forward(pointcloud_A, pointcloud_B, #points
                             features_A, features_B, # feats 
                             normals_A, normals_B, # normals
                             torch.eye(3,device = device), 0*torch.ones(3, device = device), # transform
                             pointcloud_A)
        results['scale_A'], results['scale_B'] = scale_A, scale_B
        return results
    
    def solve(self, results, transform_mode = "sim3", distance_threshold = 0.05):
        assert transform_mode in ("se3", "sim3")
        with_scaling = transform_mode == "sim3"
        src_pcd = to_o3d_pcd(to_array(results['src_corr_points']))
        tgt_pcd = to_o3d_pcd(to_array(results['tgt_corr_points']))
        correspondences = torch.from_numpy(np.arange(results['src_corr_points'].shape[0])[:, np.newaxis]).expand(-1, 2)
        correspondences = o3d.utility.Vector2iVector(to_array(correspondences))
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src_pcd, 
            tgt_pcd,
            correspondences,
            distance_threshold, # distance threshold (note: this is a bit problematic when dealing with scaling, but no other way to do)
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling = with_scaling), # if se3 or sim3
            ransac_n = 3, # num corrs in minimal sample
            criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(50_000, 1000))
        print(result_ransac)
        T = np.copy(result_ransac.transformation)
        T[:3,:3] = results['scale_B']/results['scale_A'] * T[:3,:3]
        T[:3,3] = results['scale_B'] * T[:3,3]
        np.save("transformation", T)
        results['transformation'] = T
        return results

def calibrate_neighbors(dataset, config, keep_ratio=0.8, samples_threshold=2000):
    # timer = Timer()
    # last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in tqdm.tqdm(range(len(dataset))):
        # timer.tic()
        # batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)
        inputs = dataset[i]
        batched_input = [[
                torch.from_numpy(inputs[0]).cpu().float(),#src_pcd
                torch.from_numpy(inputs[1]).cpu().float(),#tgt_pcd
                torch.from_numpy(inputs[4]).cpu().float(),#src_feats
                torch.from_numpy(inputs[5]).cpu().float(),#tgt_feats
                inputs[6],                                #rot
                inputs[7],                                #trans
                torch.ones(1).long(),                     #matching_inds,
                inputs[0],                                #src_pcd_raw
                inputs[1],                                #tgt_pcd_raw
                torch.ones(1)                             #sample
                        ]]
        batched_input = collate_fn_descriptor(batched_input, config, [hist_n] * 5)
        
        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        # timer.toc()

        # if timer.total_time - last_display > 0.1:
        #     last_display = timer.total_time
        #     print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits

def calibrate_neighbors_cambridge(
    config, keep_ratio=0.8, samples_threshold=2000
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
    method = "sift-sattler"#"sp+sg"#"sift-sattler" #"sfm_disk+lightglue" #
    scenes = ["ShopFacade", "GreatCourt", "KingsCollege", "OldHospital", "StMarysChurch"]
    print("Doing cambridge benchmark with SE(3) (shared_scale = True), set shared_scale = False if you want Sim3 benchmark")
    
    # Get histogram of neighborhood sizes i in 1 epoch max.
    import tqdm
    for scene in tqdm.tqdm(scenes):
        points_A = np.load(f"data/sfmreg/cambridge/{scene}_benchmark/pointclouds/train/{method}_cloud.npy")
        viewpoints_A = np.load(f"data/sfmreg/cambridge/{scene}_benchmark/pointclouds/train/{method}_viewpoints.npy")
        rot_ab = np.eye(3)
        tr_ab = np.zeros([3])
        if False:
            from scipy.spatial.transform import Rotation
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            points_A = np.ascontiguousarray((np.matmul(rot_ab,points_A.T).T))
            viewpoints_A = np.ascontiguousarray((np.matmul(rot_ab,viewpoints_A.T).T))
        points_B = np.load(f"data/sfmreg/cambridge/{scene}_benchmark/pointclouds/test/{method}_cloud.npy")
        viewpoints_B = np.load(f"data/sfmreg/cambridge/{scene}_benchmark/pointclouds/test/{method}_viewpoints.npy")
        
        batched_input = [[
                torch.from_numpy(points_A).float(),#src_pcd
                torch.from_numpy(points_B).float(),#tgt_pcd
                torch.ones([len(points_A), 1]).float(),#src_feats
                torch.ones([len(points_B), 1]).float(),#tgt_feats
                np.eye(3, dtype=np.float32), #rot
                np.zeros([3], dtype=np.float32), #trans
                torch.ones(1).long(), #matching_inds,
                points_A,#src_pcd_raw
                points_B,#tgt_pcd_raw
                torch.ones(1)                             #sample
                        ]]
        batched_input = collate_fn_descriptor(batched_input, config, [hist_n] * 5)

        gc.collect()
        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits