import pycolmap
import numpy as np
import torch
import os
import json
import pickle as pkl
from copy import copy



def estimate_scale(pointcloud):
    pointcloud_mean = pointcloud.mean(dim=0,keepdim=True)
    corrs = (pointcloud-pointcloud_mean).T @ (pointcloud-pointcloud_mean) / len(pointcloud)
    biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().float()
    scale = biggest_singular / np.sqrt(2)
    return scale.item()

from pathlib import Path
dataset_dir = Path("datasets/sfmreg/megadepth")

pointcloud_folder = "pointclouds_megadepth"
versions = ["trajectories1", "trajectories2", "trajectories3", "random"]
methods = ["sift", "sosnet"]
pts = {}
all_overlaps = []
for version in versions:
    pointcloud_dir = dataset_dir / pointcloud_folder / f"{version}"
    sfm_dir = dataset_dir / version / f"reconstructions/sfm"
    scenes = os.listdir(sfm_dir)
    for method in methods:
        for scene in np.random.permutation(scenes):
            os.makedirs(f"{pointcloud_dir}/{scene}", exist_ok=True)
            if os.path.exists(f"{pointcloud_dir}/{scene}/{method}_infos"):
                print(f"{scene}/{method}_infos exists, continuing")
                continue
            partial_reconstructions = [traj for traj in os.listdir(f"{sfm_dir}/{scene}")]
            remaining_reconstruction = copy(partial_reconstructions)
            for traj in partial_reconstructions:
                try:
                    from scipy.spatial.transform import Rotation as R
                    model = pycolmap.Reconstruction(f"{sfm_dir}/{scene}/{traj}/{method}")
                    pt = np.array([pt.xyz for pt in model.points3D.values()])
                    if len(pt) < 1000:
                        print(f"Too few points in {scene}, {traj} with {len(pt)} points")
                        remaining_reconstruction.remove(traj)
                        continue
                    pts[traj] = pt 
                    viewpoints = np.array([model.images[pt.track.elements[0].image_id].projection_center() for pt in model.points3D.values()])
                    # enforce at least 1000 points
                    rgb = np.array([pt.color for pt in model.points3D.values()])
                    np.save(f"{pointcloud_dir}/{scene}/{method}_colors_{traj}.npy",rgb)
                    np.save(f"{pointcloud_dir}/{scene}/{method}_cloud_{traj}.npy",pts[traj])
                    np.save(f"{pointcloud_dir}/{scene}/{method}_viewpoints_{traj}.npy", viewpoints)
                except:
                    remaining_reconstruction.remove(traj)
            partial_reconstructions = remaining_reconstruction
            names = {traj:f"{version}/{scene}/{method}_cloud_{traj}.npy" for traj in remaining_reconstruction}
            if len(names) == 0:
                print(f"No successful reconstructions for scene {scene}")
                continue
            avg_num_3D_points = sum(pt.shape[0] for pt in pts.values())/len(pts)
            if avg_num_3D_points < 1000:
                print(f"Scene {scene} has  {avg_num_3D_points=}, should investigate")
            N = len(names)
            IOU_mat = {}
            for i in partial_reconstructions:
                for j in partial_reconstructions:
                    if (j,i) in IOU_mat:
                        continue
                    if i == j:
                        continue
                    assert len(pts[i]) > 1000
                    assert len(pts[j]) > 1000
                    pts1 = torch.from_numpy(pts[i])[::5].cuda()
                    pts2 = torch.from_numpy(pts[j])[::5].cuda()
                    scale = estimate_scale(pts2) # estimate scale in the target to make sure 0.1 holds
                    pts1 = pts1 / scale
                    pts2 = pts2 / scale

                    D = torch.cdist(pts1, pts2)
                    cross_dist12, inds_12 = D.min(dim = 1)
                    cross_dist21, inds_21 = D.min(dim = 0)
                    overlap_12 = cross_dist12 < 0.1
                    overlap_21 = cross_dist21 < 0.1

                    IOU = np.sqrt((overlap_12.sum().item() / len(pts1)) * (overlap_21.sum().item() / len(pts2))) # Geometric mean of overlaps
                    IOU_mat[(i,j)] = IOU
            if len(IOU_mat.values()) == 0:
                continue 
            if max(IOU_mat.values()) < 0.3:
                print(f"Scene {scene} has max overlap {max(IOU_mat.values())}, should investigate")
            overlapping = [key for key, iou in IOU_mat.items() if iou > 0]
            all_overlaps += [IOU_mat[i, j].item() for i,j in overlapping]
            metadata = {"src": [names[i] for i,j in overlapping],
                        "tgt": [names[j] for i,j in overlapping],
                        "overlaps": [IOU_mat[i, j] for i,j in overlapping], 
                        "rot": torch.eye(3)[None].expand(len(overlapping),3,3).tolist(),
                        "trans": torch.zeros(3)[None].expand(len(overlapping),3).tolist()}
            pkl.dump(metadata, open(f"{pointcloud_dir}/{scene}/{method}_infos","wb"))