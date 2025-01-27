from typing import Any, Mapping
import warnings
import numpy as np
import torch, time, gc
from torch import nn
import open3d as o3d

from colabsfm.utils import get_correspondences, estimate_normals, transform
from colabsfm.benchmarks.benchmark_utils import to_array, to_o3d_pcd
from colabsfm.pareconv.utils.data import registration_collate_fn_stack_mode, precompute_neibors
from colabsfm.pareconv.utils.torch import to_cuda, release_cuda


def estimate_scale(pointcloud):
    pointcloud_mean = pointcloud.mean(dim=0,keepdim=True)
    corrs = (pointcloud-pointcloud_mean).T @ (pointcloud-pointcloud_mean) / len(pointcloud)
    biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().float()
    scale = biggest_singular / np.sqrt(2)
    return scale

class PareNet(nn.Module):
    def __init__(self, model: nn.Module, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.config = config
        self.model.eval()
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return self.model.load_state_dict(state_dict, strict, assign)

    @torch.inference_mode()
    def register(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None, 
                 viewpoints_A = None, viewpoints_B = None, shared_scale = False):
        pointcloud_A, pointcloud_B = pointcloud_A.float(), pointcloud_B.float()
        if len(pointcloud_A) > self.config.num_points:
            warnings.warn(f"Pointcloud A larger than {self.config.num_points} samples, downsampling to {self.config.num_points}")
            inds_A = np.random.choice(len(pointcloud_A), size = self.config.num_points, replace = False)
            pointcloud_A = pointcloud_A[inds_A]
            viewpoints_A = viewpoints_A[inds_A] if viewpoints_A is not None else None
            normals_A = normals_A[inds_A] if normals_A is not None else None
        if len(pointcloud_B) > self.config.num_points:
            warnings.warn("Pointcloud B larger than {self.config.num_points} samples, downsampling to {self.config.num_points}")
            inds_B = np.random.choice(len(pointcloud_B), size = self.config.num_points, replace = False)
            pointcloud_B = pointcloud_B[inds_B]
            viewpoints_B = viewpoints_B[inds_B] if viewpoints_B is not None else None
            normals_B = normals_B[inds_B] if normals_B is not None else None


        results = self.match(pointcloud_A, pointcloud_B, 
                            #  normals_A = normals_A, normals_B = normals_B, 
                            #  viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B, 
                             shared_scale = shared_scale)
        print("errors scaled, no transform", ((results['src_corr_points'] - results['tgt_corr_points']).norm(dim=-1) < 0.1).float().mean())
        print(len(results['src_corr_points']), "num corrs")
        results = self.solve(results, transform_mode = "se3")
        return results

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, 
                src_normals=None, tgt_normals=None, rot=None, trans=None, src_raw_pcd=None):
        
        data_dict = {
            "ref_points": tgt_pcd.cpu().float(),
            "src_points": src_pcd.cpu().float(),
            "ref_feats": tgt_feats.cpu().float(),
            "src_feats": src_feats.cpu().float(),
            "transform": torch.eye(4)
        }
        del src_pcd, tgt_pcd, src_feats, tgt_feats
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)
        
        print(data_dict['ref_points'].shape, data_dict['src_points'].shape)
        
        if self.config.neighbors is None:
            raise ValueError("Not found neighbors.")
        
        if self.config.debug: 
            print("Shapes", data_dict['ref_points'].shape, data_dict['ref_points'].dtype, data_dict['src_points'].shape, data_dict['src_points'].dtype)
            
        t0 = time.time()
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], 
            self.config.num_stages, 
            self.config.init_voxel_size,
            self.config.neighbors,
            self.config.subs_ratio
        ) 
        if self.config.debug: 
            print("Collate spent %.4f" % (time.time() - t0))
            print("Voxelization", self.config.num_stages, self.config.init_voxel_size, self.config.subs_ratio, data_dict['lengths'])
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)

        # This IS fast and optimal
        data_dict = to_cuda(data_dict)
        
        data = precompute_neibors(data_dict['points'], data_dict['lengths'],
                                  self.config.num_stages, 
                                  self.config.neighbors,
                                )
        data_dict.update(data)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)

        results_ = self.model(data_dict)
        data_dict = release_cuda(data_dict)
        results_ = release_cuda(results_)
        del data_dict

        torch.cuda.empty_cache()

        results = dict()
        for k,v in results_.items():
            if "corr" in k or k == "matching_scores" or k[-2:] == "_c": 
                nam = k.replace("ref", "tgt")
            else: nam = k
            if isinstance(v, np.ndarray): results[nam] = torch.from_numpy(v)
            else: results[nam] = torch.tensor(v)
        del results_
        gc.collect()
        torch.cuda.empty_cache()
        
        results["tgt_node_feats"] = results.pop("tgt_feats_c")
        results["src_node_feats"] = results.pop("src_feats_c")
        results["tgt_nodes"] = results.pop("tgt_points_c")
        results["src_nodes"] = results.pop("src_points_c")

        match_ = results['matching_scores'].new_zeros([results['matching_scores'].shape[0],
                                                       results['matching_scores'].shape[1] + 1,
                                                       results['matching_scores'].shape[2] + 1])
        match_[:, :-1, :-1] = results['matching_scores']
        results['matching_scores'] = match_

        return results

    @torch.inference_mode()
    def match(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None, 
              viewpoints_A = None, viewpoints_B = None, features_A = None, features_B = None, 
              shared_scale = False, voxel_downsample = False, voxel_size = 0.02, device = "cuda"):
        self.train(False)
        
        scale_A = estimate_scale(pointcloud_A).item()
        pointcloud_A = pointcloud_A / scale_A
        if shared_scale:
            scale_B = scale_A
        else:
            scale_B = estimate_scale(pointcloud_B).item()
        pointcloud_B = pointcloud_B / scale_B
        
        if voxel_downsample:
            raise NotImplementedError
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
        
        # if normals_A is None or normals_B is None:
        #     normals_A = estimate_normals(pointcloud_A, view_point=viewpoints_A).float()
        #     normals_B = estimate_normals(pointcloud_B, view_point=viewpoints_B).float()
        

        results = self.forward(pointcloud_A, pointcloud_B, features_A, features_B)
        # results = self.forward(pointcloud_A, pointcloud_B, #points
        #                      features_A, features_B, # feats 
        #                     #  normals_A, normals_B, # normals
        #                      torch.eye(3,device = device), torch.zeros(3, device = device), # transform
        #                      pointcloud_A)
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

# def calibrate_neighbors_stack_mode(
#     dataset, num_stages, voxel_size, search_radius, subs_ratio, keep_ratio=0.8, sample_threshold=2000
# ):
#     # Compute higher bound of neighbors number in a neighborhood
#     hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
#     neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
#     max_neighbor_limits = [hist_n] * num_stages

#     # Get histogram of neighborhood sizes i in 1 epoch max.
#     import tqdm
#     for i in tqdm.tqdm(range(len(dataset))):
#         inputs = dataset[i]        

#         norm = False
#         thr = torch.ones([3]).cpu().float()
#         if norm:
#             t = torch.cat((torch.from_numpy(inputs[1]), 
#                            torch.from_numpy(inputs[0])), 0).squeeze()
#             thr = ((t.max(0).values - t.min(0).values) / 2).cpu().float()

#         data_dict = {
#             "ref_points": torch.from_numpy(inputs[1]).cpu().float() ,
#             "src_points": torch.from_numpy(inputs[0]).cpu().float() ,
#             "ref_feats": torch.from_numpy(inputs[5]).cpu().float() ,
#             "src_feats": torch.from_numpy(inputs[4]).cpu().float() ,
#         }

#         # data_dict = to_cuda(data_dict)
#         data_dict = registration_collate_fn_stack_mode(
#             [data_dict], num_stages, voxel_size, max_neighbor_limits, subs_ratio
#         )
#         data_dict = to_cuda(data_dict)
#         data = precompute_neibors(data_dict['points'], data_dict['lengths'],
#                                   num_stages, 
#                                   max_neighbor_limits,
#                                 )
#         data_dict.update(data)
#         data_dict = release_cuda(data_dict)
#         gc.collect()
#         # update histogram
#         counts = [np.sum(neighbors < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
#         del data_dict
#         hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
#         neighbor_hists += np.vstack(hists)

#         print(np.min(np.sum(neighbor_hists, axis=1)))
#         if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
#             break

#     cum_sum = np.cumsum(neighbor_hists.T, axis=0)
#     neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

#     return neighbor_limits

# def calibrate_neighbors_cambridge(
#     method, num_stages, voxel_size, search_radius, subs_ratio, keep_ratio=0.8, sample_threshold=2000, num_points=30000
# ):
#     # Compute higher bound of neighbors number in a neighborhood
#     hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
#     neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
#     max_neighbor_limits = [hist_n] * num_stages
#     scenes = ["ShopFacade", "GreatCourt", "KingsCollege", "OldHospital", "StMarysChurch"]
#     print("Doing cambridge benchmark with SE(3) (shared_scale = True), set shared_scale = False if you want Sim3 benchmark")
    
#     # Get histogram of neighborhood sizes i in 1 epoch max.
#     import tqdm
#     for scene in tqdm.tqdm(scenes):
#         points_A = np.load(f"data/sfmreg/cambridge/{scene}_benchmark/pointclouds/train/{method}_cloud.npy")
#         points_B = np.load(f"data/sfmreg/cambridge/{scene}_benchmark/pointclouds/test/{method}_cloud.npy")
        
#         points_A = points_A / estimate_scale(torch.from_numpy(points_A)).item()
#         points_B = points_B / estimate_scale(torch.from_numpy(points_B)).item()
        
#         if len(points_A) > num_points:
#             inds_A = np.random.choice(len(points_A), size = num_points, replace = False)
#             points_A = points_A[inds_A]
        
#         if len(points_B) > num_points:
#             inds_B = np.random.choice(len(points_B), size = num_points, replace = False)
#             points_B = points_B[inds_B]
        
#         data_dict = {
#             "ref_points": torch.from_numpy(points_B).float(),
#             "src_points": torch.from_numpy(points_A).float(),
#             "ref_feats": torch.ones((len(points_B), 1)).float(),
#             "src_feats": torch.ones((len(points_A), 1)).float(),
#             "transform": torch.eye(4)
#         }

#         data_dict = registration_collate_fn_stack_mode(
#             [data_dict], num_stages, voxel_size, max_neighbor_limits, subs_ratio
#         )
#         data_dict = to_cuda(data_dict)
#         data = precompute_neibors(data_dict['points'], data_dict['lengths'],
#                                   num_stages, 
#                                   max_neighbor_limits,
#                                 )
#         data_dict.update(data)
#         data_dict = release_cuda(data_dict)
#         gc.collect()
#         torch.cuda.empty_cache()
#         # update histogram
#         counts = [np.sum(neighbors < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
#         del data_dict
#         hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
#         neighbor_hists += np.vstack(hists)

#         if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
#             break

#     cum_sum = np.cumsum(neighbor_hists.T, axis=0)
#     neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

#     return neighbor_limits
