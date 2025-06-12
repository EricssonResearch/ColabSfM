import os,sys,glob,torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from colabsfm.utils import to_tsfm, get_correspondences, vis_pointcloud_matplotlib, to_o3d_pcd, normal_redirect, random_cut_pointclouds
import open3d as o3d
from time import perf_counter
import pickle as pkl

class ColabSfM(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self, infos, config, data_root = "data", pose_info = None, data_augmentation=False, overlap_threshold = 0.3):
        super().__init__()
        enough_overlap = np.array(infos["overlaps"]) > overlap_threshold
        infos = {k:[item for i, item in enumerate(v) if enough_overlap[i]] for k,v in infos.items()}
        #debug = np.array(["trajectories1/0190/sift_cloud_trajectory_10.npy" in src for src in infos["src"]])
        #infos = {k:[item for i, item in enumerate(v) if debug[i]] for k,v in infos.items()}
        self.infos = infos
        self.base_dir = config.root
        #self.overlap_radius = config.overlap_radius
        self.data_augmentation=data_augmentation
        self.config = config
        self.data_root = data_root
        
        self.rot_factor=1.
        self.augment_noise = config.augment_noise
        self.max_points = 16_000 if self.config.mode == "train" else 30_000 # TODO: make 30_000 again
        self.view_point = np.array([0., 0., 0.])
        self.mode = config.colabsfm_mode # aligned, sim3
        self.pose_info = pose_info
        self.low_overlap = config.low_overlap

    def __len__(self):
        return len(self.infos['rot'])
    
    def __getitem__(self,item):
        t0 = perf_counter()
        # get transformation
        rot=np.array(self.infos['rot'][item])
        trans=np.array(self.infos['trans'][item])

        # get pointcloud
        src_path=os.path.join(self.base_dir,self.infos['src'][item])
        tgt_path=os.path.join(self.base_dir,self.infos['tgt'][item])
        src_pcd = np.load(src_path)
        tgt_pcd = np.load(tgt_path)
        
        if self.config.normal_orientation == "hoppe":
            raise ValueError("Don't use hoppe")
        elif self.config.normal_orientation == "viewpoint":
            src_viewpoint_path = src_path.replace("_cloud_", "_viewpoints_")
            tgt_viewpoint_path = tgt_path.replace("_cloud_", "_viewpoints_")
            src_viewpoints = np.load(src_viewpoint_path)
            tgt_viewpoints = np.load(tgt_viewpoint_path)


        if self.config.use_color or self.config.use_gray:
            src_feats_path = src_path.replace("_cloud_", "_colors_")
            tgt_feats_path = tgt_path.replace("_cloud_", "_colors_")
            src_feats = np.load(src_feats_path).astype(np.float32)/255.
            tgt_feats = np.load(tgt_feats_path).astype(np.float32)/255.
            if self.config.use_gray:
                src_feats = np.mean(src_feats, axis = 1, keepdims=True)
                tgt_feats = np.mean(tgt_feats, axis = 1, keepdims=True)

        else:
            src_feats = np.ones_like(src_pcd[:,:1]).astype(np.float32)
            tgt_feats = np.ones_like(tgt_pcd[:,:1]).astype(np.float32)
        
        if True:            
            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd=np.matmul(rot_ab,src_pcd.T).T
                src_viewpoints=np.matmul(rot_ab,src_viewpoints.T).T
                rot=np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd=np.matmul(rot_ab,tgt_pcd.T).T
                tgt_viewpoints=np.matmul(rot_ab,tgt_viewpoints.T).T
                rot=np.matmul(rot_ab,rot)
                trans=np.matmul(rot_ab,trans)

        o3d_src_pcd = to_o3d_pcd(src_pcd)
        o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
        
        o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        src_normals = np.asarray(o3d_src_pcd.normals)
        src_pcd_normals = normal_redirect(src_pcd, src_normals, view_point=src_viewpoints)


        o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        tgt_normals = np.asarray(o3d_tgt_pcd.normals)
        tgt_pcd_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=tgt_viewpoints)
        


        if self.low_overlap and (self.infos['overlaps'][item] > 0.8) and (len(src_pcd) > 2000) and (len(tgt_pcd) > 2000):
            src_inds, tgt_inds = random_cut_pointclouds(src_pcd, tgt_pcd)
            src_pcd = src_pcd[src_inds]
            tgt_pcd = tgt_pcd[tgt_inds]
            src_feats = src_feats[src_inds]
            tgt_feats = tgt_feats[tgt_inds]

            src_pcd_normals = src_pcd_normals[src_inds]
            tgt_pcd_normals = tgt_pcd_normals[tgt_inds]


        # if we get too many points, we do some downsampling
        if(src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
            src_feats = src_feats[idx]
            src_pcd_normals = src_pcd_normals[idx]
        if(tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]
            tgt_feats = tgt_feats[idx]
            tgt_pcd_normals = tgt_pcd_normals[idx]


        #Scale pointclouds
        tgt_pcd = torch.tensor(tgt_pcd).float()
        # We choose scale such that the resulting standard deviation is about 1.44 in the major axis of the pointcloud
        # We will use 0.1 as a threshold for a good match later,
        # assuming that the pointcloud is uniformly distributed this means that we have an error of about 2% of the "size" of the pointcloud
        
        tgt_pcd_mean = tgt_pcd.mean(dim=0,keepdim=True)
        corrs = (tgt_pcd-tgt_pcd_mean).T @ (tgt_pcd-tgt_pcd_mean) / len(tgt_pcd)
        biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().numpy()
        tgt_scale = biggest_singular / np.sqrt(2) 
        tgt_pcd = tgt_pcd / tgt_scale

        src_pcd = torch.tensor(src_pcd).float()
        if self.mode in ("aligned", "se3"):
            src_scale = tgt_scale
        elif self.mode == "sim3":
            src_pcd_mean = src_pcd.mean(dim=0,keepdim=True)
            corrs = (src_pcd-src_pcd_mean).T @ (src_pcd-src_pcd_mean) / len(src_pcd)
            biggest_singular = torch.linalg.eigh(corrs).eigenvalues.max().sqrt().numpy()
            src_scale = biggest_singular / np.sqrt(2) 
        src_pcd = src_pcd / src_scale
        
        # Update relative transform accordingly
        trans = trans / tgt_scale
        rot = (src_scale / tgt_scale) * rot
        # print(item,'scales',src_scale,tgt_scale,src_scale / tgt_scale)

        src_pcd_normals = torch.tensor(src_pcd_normals).float()
        tgt_pcd_normals = torch.tensor(tgt_pcd_normals).float()
        #print(trans,src_pcd)
        return src_pcd.numpy().astype(np.float32), tgt_pcd.numpy().astype(np.float32), \
            src_pcd_normals.numpy().astype(np.float32), tgt_pcd_normals.numpy().astype(np.float32),\
            src_feats, tgt_feats,\
            rot.astype(np.float32), trans.astype(np.float32),\
            src_pcd.numpy().astype(np.float32), None, src_path, tgt_path