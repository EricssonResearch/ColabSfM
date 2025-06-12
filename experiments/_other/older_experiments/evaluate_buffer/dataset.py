import torch.utils.data as Data
import os
import open3d as o3d
import glob
import numpy as np
from utils.SE3 import *
from utils.common import make_open3d_point_cloud
import torch

def convert_pointcloud(xyz0, xyz1, config):
    src_pcd = make_open3d_point_cloud(xyz0, [1, 0.706, 0])
    src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=config.data.downsample)
    src_pts = np.array(src_pcd.points)
    np.random.shuffle(src_pts)

    tgt_pcd = make_open3d_point_cloud(xyz1, [0, 0.651, 0.929])
    tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=config.data.downsample)

    relt_pose = np.eye(4, dtype = np.float32)

    tgt_pts = np.array(tgt_pcd.points)
    np.random.shuffle(tgt_pts)

    # second sample
    ds_size = config
    src_pcd = o3d.geometry.PointCloud.voxel_down_sample(src_pcd, voxel_size=ds_size)
    src_kpt = np.array(src_pcd.points)
    np.random.shuffle(src_kpt)

    tgt_pcd = o3d.geometry.PointCloud.voxel_down_sample(tgt_pcd, voxel_size=ds_size)
    tgt_kpt = np.array(tgt_pcd.points)
    np.random.shuffle(tgt_kpt)


    src_pcd = make_open3d_point_cloud(src_kpt, [1, 0.706, 0])
    src_pcd.estimate_normals()
    src_pcd.orient_normals_towards_camera_location()
    src_noms = np.array(src_pcd.normals)
    src_kpt = np.concatenate([src_kpt, src_noms], axis=-1)

    tgt_pcd = make_open3d_point_cloud(tgt_kpt, [0, 0.651, 0.929])
    tgt_pcd.estimate_normals()
    tgt_pcd.orient_normals_towards_camera_location()
    tgt_noms = np.array(tgt_pcd.normals)
    tgt_kpt = np.concatenate([tgt_kpt, tgt_noms], axis=-1)

    return {'src_fds_pts': src_pts,  # first downsampling
            'tgt_fds_pts': tgt_pts,
            'relt_pose': relt_pose,
            'src_sds_pts': src_kpt,  # second downsampling
            'tgt_sds_pts': tgt_kpt}
