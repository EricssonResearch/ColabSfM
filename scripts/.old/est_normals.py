import numpy as np
from colabsfm.utils import to_o3d_pcd
import open3d as o3d

src_pcd = np.random.randn(1000, 3)
o3d_src_pcd = to_o3d_pcd(src_pcd)
o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
