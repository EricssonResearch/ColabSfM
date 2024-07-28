import numpy as np
import open3d as o3d

o3d_src_pcd = to_o3d_pcd(src_pcd)
o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
src_normals = np.asarray(o3d_src_pcd.normals)
src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
tgt_normals = np.asarray(o3d_tgt_pcd.normals)
tgt_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=self.view_point)
src_feats = np.ones(shape=(src_pcd.shape[0], 1))
tgt_feats = np.ones(shape=(tgt_pcd.shape[0], 1))
