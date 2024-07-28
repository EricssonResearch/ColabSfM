from pathlib import Path
import pycolmap
import numpy as np
import os

N = 10
method = "sift"
pts = []
scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
for scene in scenes:
    sfm_dir = Path(f"datasets/7scenes/7scenes_sfm_triangulated/{scene}")
    pointcloud_dir = f"datasets/sfmreg/7scenes/{scene}_benchmark/pointclouds"
    for split in ["Train", "Test"]:#scenes:
        try:
            os.makedirs(f"{pointcloud_dir}/{split}", exist_ok=True)
        except:
            continue
        model = pycolmap.Reconstruction(sfm_dir / split / method)
        pts = np.array([pt.xyz for pt in model.points3D.values()])
        color = np.array([pt.color for pt in model.points3D.values()])
        viewpoints = np.array([model.images[pt.track.elements[0].image_id].projection_center() for pt in model.points3D.values()])
        np.save(f"{pointcloud_dir}/{split}/{method}_cloud.npy",pts)
        np.save(f"{pointcloud_dir}/{split}/{method}_colors.npy", color)
        np.save(f"{pointcloud_dir}/{split}/{method}_viewpoints.npy", viewpoints)
