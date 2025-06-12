import warnings
import numpy as np
import torch
from torch import nn
import open3d as o3d

from colabsfm.utils import get_correspondences, estimate_normals, transform
from colabsfm.benchmarks.benchmark_utils import to_array, to_o3d_pcd

class RoITr(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
    
    @torch.inference_mode()
    def register(self, pointcloud_A:torch.Tensor, pointcloud_B:torch.Tensor, normals_A = None, normals_B = None, viewpoints_A:np.ndarray = None, viewpoints_B:np.ndarray = None, features_A = None, features_B = None, shared_scale = False):
        pointcloud_A, pointcloud_B = pointcloud_A.float(), pointcloud_B.float()
        if len(pointcloud_A) > 30_000:
            # warnings.warn("Pointcloud A (%d) larger than 30_000 samples, downsampling to 30_000" % len(pointcloud_A))
            inds_A = np.random.choice(len(pointcloud_A), size = 30_000, replace = False)
            pointcloud_A = pointcloud_A[inds_A]
            viewpoints_A = viewpoints_A[inds_A] if viewpoints_A is not None else None
            normals_A = normals_A[inds_A] if normals_A is not None else None
        if len(pointcloud_B) > 30_000:
            # warnings.warn("Pointcloud B (%d) larger than 30_000 samples, downsampling to 30_000" % len(pointcloud_B))
            inds_B = np.random.choice(len(pointcloud_B), size = 30_000, replace = False)
            pointcloud_B = pointcloud_B[inds_B]
            viewpoints_B = viewpoints_B[inds_B] if viewpoints_B is not None else None
            normals_B = normals_B[inds_B] if normals_B is not None else None


        self.train(False)
        results = self.match(pointcloud_A, pointcloud_B, 
                             normals_A = normals_A, normals_B = normals_B, 
                             viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B,
                             features_A = features_A, features_B = features_B,
                             shared_scale = shared_scale)
        # print("errors scaled, no transform", ((results['src_corr_points'] - results['tgt_corr_points']).norm(dim=-1) < 0.1).float().mean())
        # print(len(results['src_corr_points']), "num corrs")
        results = self.solve(results, transform_mode = "se3")
        return results

    def estimate_scale(self, pointcloud):
        pointcloud_mean = pointcloud.mean(dim=0,keepdim=True)
        corrs = (pointcloud-pointcloud_mean).T @ (pointcloud-pointcloud_mean) / len(pointcloud)
        biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().float()
        scale = biggest_singular / np.sqrt(2)
        return scale

    @torch.inference_mode()
    def match(self, pointcloud_A:torch.Tensor, pointcloud_B:torch.Tensor, normals_A = None, normals_B = None, 
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
        results = self.model(pointcloud_A, pointcloud_B, #points
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
        T = np.copy(result_ransac.transformation)
        T[:3,:3] = results['scale_B']/results['scale_A'] * T[:3,:3]
        T[:3,3] = results['scale_B'] * T[:3,3]
        # np.save("transformation", T)
        results['transformation'] = T
        return results
    