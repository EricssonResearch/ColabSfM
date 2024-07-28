import pycolmap
import numpy as np
import torch
import os
import json
import pickle as pkl
N = 10

sfm_dir = "tum_triangulated"
pointcloud_dir = "tum_benchmark/pointclouds"
scenes = os.listdir(sfm_dir)
method = "sift"#"sfm_disk+lightglue"
pts = []
for scene in scenes:
    try:
        os.makedirs(f"{pointcloud_dir}/{scene}", exist_ok=True)
    except:
        continue
    if os.path.exists(f"{pointcloud_dir}/{scene}/{method}_infos"):
        print(f"{scene}/{method}_infos exists, continuing")
        continue
    model = pycolmap.Reconstruction(f"{sfm_dir}/{scene}/{method}")
    pts.append(np.array([pt.xyz for pt in model.points3D.values()]))
    viewpoints = np.array([model.images[pt.track.elements[0].image_id].projection_center() for pt in model.points3D.values()])
    np.save(f"{pointcloud_dir}/{scene}/{method}_cloud.npy",pts[-1])
    np.save(f"{pointcloud_dir}/{scene}/{method}_viewpoints.npy", viewpoints)
