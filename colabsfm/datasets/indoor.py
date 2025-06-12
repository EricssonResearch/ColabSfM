"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,sys,glob,torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from colabsfm.utils import to_tsfm, get_correspondences, vis_pointcloud_matplotlib, to_o3d_pcd, normal_redirect
import open3d as o3d
from time import perf_counter

class IndoorDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,infos,config, data_root = "data", max_points = 16_000, data_augmentation=False):
        super(IndoorDataset,self).__init__()
        self.infos = infos
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.data_augmentation=data_augmentation
        self.config = config
        self.data_root = data_root
        
        self.rot_factor=1.
        self.augment_noise = config.augment_noise
        self.max_points = max_points
        self.view_point = np.array([0., 0., 0.])

    def __len__(self):
        return len(self.infos['rot'])
    
    def __getitem__(self,item):
        t0 = perf_counter()
        # get transformation
        rot=self.infos['rot'][item]
        trans=self.infos['trans'][item]

        # get pointcloud
        src_path=os.path.join(self.base_dir,self.infos['src'][item])
        tgt_path=os.path.join(self.base_dir,self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)
        normals_dir = f"{self.data_root}/indoor_normals"
        src_normals_path = os.path.join(normals_dir, self.infos['src'][item])
        tgt_normals_path = os.path.join(normals_dir, self.infos['tgt'][item])
        try:
            src_pcd_normals = torch.load(src_normals_path)
            tgt_pcd_normals = torch.load(tgt_normals_path)
        except:
            o3d_src_pcd = to_o3d_pcd(src_pcd)
            o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
            o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
            src_normals = np.asarray(o3d_src_pcd.normals)
            src_pcd_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
            o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
            tgt_normals = np.asarray(o3d_tgt_pcd.normals)
            tgt_pcd_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=self.view_point)

        if False:
            o3d_src_pcd = to_o3d_pcd(src_pcd)
            o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
            o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
            src_normals = np.asarray(o3d_src_pcd.normals)
            src_normals = normal_redirect(src_pcd, src_normals, view_point=self.view_point)
            o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
            tgt_normals = np.asarray(o3d_tgt_pcd.normals)
            tgt_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=self.view_point)
            base_dir = "data/indoor_normals"
            src_normals_path = os.path.join(base_dir, self.infos['src'][item])
            tgt_normals_path = os.path.join(base_dir, self.infos['tgt'][item])
            os.makedirs(os.path.dirname(src_normals_path), exist_ok=True)
            os.makedirs(os.path.dirname(src_normals_path), exist_ok=True)
            torch.save(src_normals, open(src_normals_path,"wb"))
            torch.save(tgt_normals, open(tgt_normals_path,"wb"))
            return "ok!"
        # if we get too many points, we do some downsampling
        if(src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
            src_pcd_normals = src_pcd_normals[idx]
        if(tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]
            tgt_pcd_normals = tgt_pcd_normals[idx]

        # add gaussian noise
        if self.data_augmentation:            
            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2/self.rot_factor # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd=np.matmul(rot_ab,src_pcd.T).T
                rot=np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd=np.matmul(rot_ab,tgt_pcd.T).T
                rot=np.matmul(rot_ab,rot)
                trans=np.matmul(rot_ab,trans)

            src_pcd += (np.random.rand(src_pcd.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0],3) - 0.5) * self.augment_noise
        
        if(trans.ndim==1):
            trans=trans[:,None]
        from colabsfm.utils import from_homogeneous, to_homogeneous

        tsfm = torch.tensor(to_tsfm(rot, trans)).float()
        if False:
            src_pcd = from_homogeneous((tsfm @ to_homogeneous(src_pcd)[...,None])[...,0])
            tsfm = torch.eye(4)
        elif False:
            #tsfm[:3,-1] = 0#torch.eye(3)
            #tsfm = torch.eye(4)
            #tgt_pcd = src_pcd
            tgt_pcd = from_homogeneous((tsfm @ to_homogeneous(src_pcd)[...,None])[...,0])#src_pcd #
            #tsfm = tsfm#torch.eye(4)#
        # get correspondence at fine level
        src_pcd = torch.tensor(src_pcd).float()
        tgt_pcd = torch.tensor(tgt_pcd).float()

        src_pcd_normals = torch.tensor(src_pcd_normals).float()
        tgt_pcd_normals = torch.tensor(tgt_pcd_normals).float()
        #print(trans,src_pcd)

        return {
                "pc_A": src_pcd, "pc_B": tgt_pcd,
                 "features_A": src_pcd,
                 "features_B": tgt_pcd,
                 "normals_A": src_pcd_normals,
                 "normals_B": tgt_pcd_normals,
                   "tsfm": tsfm,
                   "num_pt_A": len(src_pcd),
                   "num_pt_B": len(tgt_pcd),
                   "overlap": self.infos['overlap'][item]}
