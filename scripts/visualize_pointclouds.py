import pycolmap
import rerun as rr
from easy_gravity import get_gravity_from_colmap_reconstruction
import numpy as np

reconstr =pycolmap.Reconstruction("datasets/cambridge/cambridge-retriangulated/GreatCourt/test/sift")#Train/sfm")
colors = np.array([pt3d.color for pt3d in reconstr.points3D.values()])
xyzs = np.array([pt3d.xyz for pt3d in reconstr.points3D.values()])
#cams = np.array([im.])
xyzs = xyzs-xyzs.mean(axis=0, keepdims=True)
gravity = get_gravity_from_colmap_reconstruction(reconstr)
print(gravity)
rr.init("vis", spawn=True)
rr.log("/", rr.ViewCoordinates.LFU, static=True)
rr.log("world/pts3d",rr.Points3D(positions = xyzs, colors = colors))

#reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/chess/triangulated")
#reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/chess/triangulated")
pass