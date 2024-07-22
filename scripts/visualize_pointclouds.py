import pycolmap
import rerun as rr
reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/fire/Test/sfm")
colors = [pt3d.color for pt3d in reconstr.points3D.values()]
xyzs = [pt3d.xyz for pt3d in reconstr.points3D.values()]
rr.init("vis", spawn=True)
rr.log("world", rr.ViewCoordinates.RDF, static=True)
rr.log("world/pts3d",rr.Points3D(positions = xyzs, colors = colors))

#reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/chess/triangulated")
#reconstr =pycolmap.Reconstruction("datasets/7scenes/7scenes_sfm_triangulated/chess/triangulated")
pass