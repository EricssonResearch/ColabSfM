import pycolmap
import numpy as np
import torch
import os
import json
import pickle as pkl
N = 10

sfm_dir = "datasets/sfmreg/reconstructions/sfm"
pointcloud_dir = "datasets/sfmreg/pointclouds"
scenes = os.listdir(sfm_dir)
method = "sift"#sfm_disk+lightglue"
overlaps = []
for scene in scenes:
    try:
        os.makedirs(f"{pointcloud_dir}/{scene}", exist_ok=True)
    except:
        continue
    partial_reconstructions = [int(i) for i in os.listdir(f"{sfm_dir}/{scene}")]
    images = []
    for i in list(sorted(partial_reconstructions)[:2]):
        try:
            from scipy.spatial.transform import Rotation as R
            from hloc.visualization import visualize_sfm_2d
            from pathlib import Path
            model = pycolmap.Reconstruction(f"{sfm_dir}/{scene}/{i}/{method}")
            images.append(list(model.images.keys()))
        except:
            partial_reconstructions.remove(i)
    if len(images) > 1:
        overlaps.append(len(set(images[0]).intersection(set(images[1])))/len(set(images[0])))
    names = {i:f"{scene}/{method}_cloud_{i}.npy" for i in partial_reconstructions}
    if len(names) == 0:
        print("No successful reconstructions, continuing")
        continue
    continue
    IOU_mat = torch.zeros(N,N)
    for i in partial_reconstructions:
        for j in partial_reconstructions:
            if i >= j:
                continue
            pts1 = torch.from_numpy(pts[i])[::5].cuda()
            pts2 = torch.from_numpy(pts[j])[::5].cuda()

            D = torch.cdist(pts1, pts2)
            self_dist1 = torch.cdist(pts1, pts1).topk(k = 2, dim = 1, largest = False).values[:,-1]
            self_dist2 = torch.cdist(pts2, pts2).topk(k = 2, dim = 1, largest = False).values[:,-1]
            cross_dist12, inds_12 = D.min(dim = 1)
            cross_dist21, inds_21 = D.min(dim = 0)
            cmp_dist_2 = self_dist2[inds_12]
            cmp_dist_1 = self_dist1[inds_21]
            overlap_12 = (2*cmp_dist_2 - cross_dist12) > 0
            overlap_21 = (2*cmp_dist_1 - cross_dist21) > 0

            IOU = (overlap_12.sum() + overlap_21.sum()) / (len(pts1) + len(pts2))
            print(IOU,i,j)
            IOU_mat[i,j] = IOU
    overlapping = torch.nonzero(IOU_mat)
    metadata = {"src": [names[i.item()] for i in overlapping[:,0]],
                "tgt": [names[i.item()] for i in overlapping[:,1]],
                "overlaps": IOU_mat[overlapping[:,0], overlapping[:,1]].tolist(), 
                "rot": torch.eye(3)[None].expand(len(overlapping),3,3).tolist(),
                "trans": torch.zeros(3)[None].expand(len(overlapping),3).tolist()}
    pkl.dump(metadata, open(f"{pointcloud_dir}/{scene}/{method}_infos","wb"))
