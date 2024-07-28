import rerun as rr
import numpy as np
from pathlib import Path
#cams = np.array([im.])
bench_path = Path("datasets/sfmreg/cambridge/OldHospital_benchmark/pointclouds/test")
xyzs = np.load(bench_path / "sift_cloud.npy")
colors = np.load(bench_path / "sift_colors.npy")

rr.init("vis", spawn=True)
rr.log("/", rr.ViewCoordinates.LFU, static=True)
rr.log("world/pts3d",rr.Points3D(positions = xyzs, colors = colors))

#reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/chess/triangulated")
#reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/chess/triangulated")
pass