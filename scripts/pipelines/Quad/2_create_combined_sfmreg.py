import pycolmap
import numpy as np
import torch
import os
import json
import pickle as pkl
from copy import copy



TRIANGULATE = True
sfm = "sfm_tr" if TRIANGULATE else "sfm"
def estimate_scale(pointcloud):
    pointcloud_mean = pointcloud.mean(dim=0,keepdim=True)
    corrs = (pointcloud-pointcloud_mean).T @ (pointcloud-pointcloud_mean) / len(pointcloud)
    biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().float()
    scale = biggest_singular / np.sqrt(2)
    return scale.item()

from pathlib import Path
dataset_dir = Path("/home/ejaealb/work/hloc/datasets/sfmreg/Quad/ArtsQuad_dataset/")

pointcloud_folder = "pointclouds"
method = "sift"
pts = {}
all_overlaps = []

pointcloud_dir = dataset_dir / pointcloud_folder 
sfm_dir = dataset_dir / f"reconstructions/sfm_trajectory"

partial_reconstructions = [traj for traj in os.listdir(sfm_dir)]
remaining_reconstruction = copy(partial_reconstructions)

for traj in partial_reconstructions:
    os.makedirs(f"{pointcloud_dir}", exist_ok=True)
    if os.path.exists(f"{pointcloud_dir}/{method}_infos"):
        print(f"{traj}/{method}_infos exists, continuing")
        continue
    try:
        from scipy.spatial.transform import Rotation as R
        model = pycolmap.Reconstruction(f"{sfm_dir}/{traj}/{method}")
        pt = np.array([pt.xyz for pt in model.points3D.values()])
        if len(pt) < 1000:
            print(f"Too few points in {traj} with {len(pt)} points")
            remaining_reconstruction.remove(traj)
            continue
        pts[traj] = pt 
        viewpoints = np.array([model.images[pt.track.elements[0].image_id].projection_center() for pt in model.points3D.values()])
        # enforce at least 1000 points
        rgb = np.array([pt.color for pt in model.points3D.values()])
        np.save(f"{pointcloud_dir}/{method}_colors_{traj}.npy",rgb)
        np.save(f"{pointcloud_dir}/{method}_cloud_{traj}.npy",pts[traj])
        np.save(f"{pointcloud_dir}/{method}_viewpoints_{traj}.npy", viewpoints)
    except:
        remaining_reconstruction.remove(traj)
partial_reconstructions = remaining_reconstruction
names = {traj:f"{method}_cloud_{traj}.npy" for traj in remaining_reconstruction}
if len(names) == 0:
    raise ValueError("No successful reconstructions for scene")
avg_num_3D_points = sum(pt.shape[0] for pt in pts.values())/len(pts)
if avg_num_3D_points < 1000:
    raise ValueError(f"Scene has  {avg_num_3D_points=}, should investigate")
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
    raise ValueError(f"IOU_mat.values() = {IOU_mat.values()}")
if max(IOU_mat.values()) < 0.3:
    raise ValueError(f"Scene has max overlap {max(IOU_mat.values())}, should investigate")
overlapping = [key for key, iou in IOU_mat.items() if iou > 0]
all_overlaps += [IOU_mat[i, j].item() for i,j in overlapping]
print(len(overlapping), np.mean(list(IOU_mat.values())))
metadata = {"src": [names[i] for i,j in overlapping],
            "tgt": [names[j] for i,j in overlapping],
            "overlaps": [IOU_mat[i, j] for i,j in overlapping], 
            "rot": torch.eye(3)[None].expand(len(overlapping),3,3).tolist(),
            "trans": torch.zeros(3)[None].expand(len(overlapping),3).tolist()}
pkl.dump(metadata, open(f"{pointcloud_dir}/{method}_infos","wb"))
pkl.dump(metadata, open(f"/home/ejaealb/work/sfmreg/sfmreg/roitr/configs/quad/{method}_infos.pkl","wb"))