import pycolmap
import numpy as np
import torch
import os
import json
import pickle as pkl
N = 10
scene = "GreatCourt"
method = "sift-sattler"
pts = []
scenes = ["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
for scene in scenes:
    sfm_dir = f"datasets/cambridge/{scene}_triangulated"
    pointcloud_dir = f"datasets/sfmreg/cambridge/{scene}_benchmark/pointclouds"
    for split in ["train", "test"]:#scenes:
        try:
            os.makedirs(f"{pointcloud_dir}/{split}", exist_ok=True)
        except:
            continue
        if os.path.exists(f"{pointcloud_dir}/{split}/{method}_infos"):
            print(f"{split}/{method}_infos exists, continuing")
            continue
        model = pycolmap.Reconstruction(f"{sfm_dir}/{split}/{method}")
        pts.append(np.array([pt.xyz for pt in model.points3D.values()]))
        color = np.array([pt.color for pt in model.points3D.values()])
        viewpoints = np.array([model.images[pt.track.elements[0].image_id].projection_center() for pt in model.points3D.values()])
        np.save(f"{pointcloud_dir}/{split}/{method}_cloud.npy",pts[-1])
        np.save(f"{pointcloud_dir}/{split}/{method}_colors.npy", color)
        np.save(f"{pointcloud_dir}/{split}/{method}_viewpoints.npy", viewpoints)
