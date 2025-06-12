import numpy as np
import scipy
import open3d as o3d
import torch.nn.functional as F
import pycolmap

def to_homogeneous(x):
    if isinstance(x, np.ndarray):
        ones = np.ones_like(x[...,-1:])
        return np.concatenate((x, ones), axis = -1)
    elif isinstance(x, torch.Tensor):
        ones = torch.ones_like(x[...,-1:])
        return torch.cat((x, ones), axis = -1)
        
def from_homogeneous(xh, eps = 1e-12):
    return xh[...,:-1] / (xh[...,-1:]+eps)


def save_obj(obj, path ):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


"""
Scripts for pairwise registration using different sampling methods

Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,re,sys,json,yaml,random, glob, argparse, torch, pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

_EPS = 1e-7  # To prevent division by zero


def fmr_wrt_distance(data,split,inlier_ratio_threshold=0.05):
    """
    calculate feature match recall wrt distance threshold
    """
    fmr_wrt_distance =[]
    for distance_threshold in range(1,21):
        inlier_ratios =[]
        distance_threshold /=100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0]:ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_distance.append(fmr*100)
    return fmr_wrt_distance

def fmr_wrt_inlier_ratio(data, split, distance_threshold=0.1):
    """
    calculate feature match recall wrt inlier ratio threshold
    """
    fmr_wrt_inlier =[]
    for inlier_ratio_threshold in range(1,21):
        inlier_ratios =[]
        inlier_ratio_threshold /=100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0]:ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_inlier.append(fmr*100)

    return fmr_wrt_inlier


def write_est_trajectory(gt_folder, exp_dir, tsfm_est):
    """
    Write the estimated trajectories 
    """
    scene_names=sorted(os.listdir(gt_folder))
    count=0
    for scene_name in scene_names:
        gt_pairs, gt_traj = read_trajectory(os.path.join(gt_folder,scene_name,'gt.log'))
        est_traj = []
        for i in range(len(gt_pairs)):
            est_traj.append(tsfm_est[count])
            count+=1

        # write the trajectory
        c_directory=os.path.join(exp_dir,scene_name)
        os.makedirs(c_directory)
        write_trajectory(np.array(est_traj),gt_pairs,os.path.join(c_directory,'est.log'))


def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm
    
def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def random_sample(pcd, feats, N):
    """
    Do random sampling to get exact N points and associated features
    pcd:    [N,3]
    feats:  [N,C]
    """
    if(isinstance(pcd,torch.Tensor)):
        n1 = pcd.size(0)
    elif(isinstance(pcd, np.ndarray)):
        n1 = pcd.shape[0]

    if n1 == N:
        return pcd, feats

    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)

    return pcd[choice], feats[choice]
    
def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return: 
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2) 
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs

def get_inlier_ratio(src_pcd, tgt_pcd, src_feat, tgt_feat, rot, trans, inlier_distance_threshold = 0.1):
    """
    Compute inlier ratios with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    src_feat = to_tensor(src_feat)
    tgt_feat = to_tensor(tgt_feat)
    rot, trans = to_tensor(rot), to_tensor(trans)

    results =dict()
    results['w']=dict()
    results['wo']=dict()

    if(torch.cuda.device_count()>=1):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)
    scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()

    ########################################
    # 1. calculate inlier ratios wo mutual check
    _, idx = scores.max(-1)
    dist = torch.norm(src_pcd- tgt_pcd[idx],dim=1)
    results['wo']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['wo']['inlier_ratio'] = c_inlier_ratio

    ########################################
    # 2. calculate inlier ratios w mutual check
    selection = mutual_selection(scores[None,:,:])[0]
    row_sel, col_sel = np.where(selection)
    dist = torch.norm(src_pcd[row_sel]- tgt_pcd[col_sel],dim=1)
    results['w']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['w']['inlier_ratio'] = c_inlier_ratio

    return results

def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point(s)
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column
    
    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N] 
    """
    score_mat=to_array(score_mat)
    if(score_mat.ndim==2):
        score_mat=score_mat[None,:,:]
    
    mutuals=np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]): # loop through the batch
        c_mat=score_mat[i]
        flag_row=np.zeros_like(c_mat)
        flag_column=np.zeros_like(c_mat)

        max_along_row=np.argmax(c_mat,1)[:,None]
        max_along_column=np.argmax(c_mat,0)[None,:]
        np.put_along_axis(flag_row,max_along_row,1,1)
        np.put_along_axis(flag_column,max_along_column,1,0)
        mutuals[i]=(flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)  


def get_scene_split(whichbenchmark):
    """
    Just to check how many valid fragments each scene has 
    """
    assert whichbenchmark in ['3DMatch','3DLoMatch']
    folder = f'configs/benchmarks/{whichbenchmark}/*/gt.log'

    scene_files=sorted(glob.glob(folder))
    split=[]
    count=0
    for eachfile in scene_files:
        gt_pairs, gt_traj = read_trajectory(eachfile)
        split.append([count,count+len(gt_pairs)])
        count+=len(gt_pairs)
    return split

def get_correspondences(pc_A, pc_B, tsfm, overlap_radius, mask_A = None, mask_B = None, require_mnn = True, as_tuple = True):
    pc_A_to_B = from_homogeneous((tsfm[:,None] @ to_homogeneous(pc_A)[...,None])[...,0])
    d = torch.cdist(pc_A_to_B.cuda(), pc_B.cuda())
    B,M,N = d.shape
    if mask_A is not None:
        d[~mask_A.expand(B,M,N)] = 1e10
    if mask_B is not None:
        d[~mask_B.mT.expand(B,M,N)] = 1e10
    P = (d < overlap_radius)
    if require_mnn:
        P = (d == d.min(dim=-2, keepdim = True).values) * (d == d.min(dim=-1, keepdim = True).values) * P
    corresps = (P).nonzero(as_tuple=as_tuple)
    matchability_A = P.sum(dim=-1)
    matchability_B = P.sum(dim=-2)

    return corresps, matchability_A, matchability_B


def get_best_device(verbose = False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if verbose: print (f"Fastest device found is: {device}")
    return device

def to_best_device(batch, device=get_best_device()):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch



def vis_pointcloud_matplotlib(pts, name = "pointcloud"):
    import matplotlib.pyplot as plt
    f = plt.figure()
    ax = f.add_subplot(111, projection = '3d')
    ax.scatter(pts[:, 0],
            pts[:, 2],
            pts[:, 1],
            c=pts[:, 2],
            s = 0.3,
            cmap='jet')
    plt.savefig(f"{name}.png")
    plt.close()

@torch.no_grad()
def least_squares_normal_est(delta, abs_pos):
    # x: (..., N, d)
    # abs_pos: (..., d)
    outer_prod = delta.mT @ delta
    _, Q  = torch.linalg.eigh(outer_prod)
    n_hat = Q[...,:,0]
    # We assert that the normals should face the camera
    facing = (n_hat * abs_pos).sum(dim=-1, keepdim = True).sign()
    # Hence the normal should be the opposite direction of the absolute position
    n = - n_hat * facing
    return n


@torch.no_grad()
def compute_approx_knn(x, K, num_anchors = 500, anchor_K = 500): 
    # x = (B,N,D)
    # computation time: N * num_anchors + N * anchor_K
    # memory: same kind of
    # Should not be run for N < 1000
    # Also requires too much memory if D >> 1
    B,N,D = x.shape
    anchor_stride = N//num_anchors
    M = (N//num_anchors) * num_anchors
    anchors = x[:,::anchor_stride]
    D_anch = torch.cdist(x,anchors)
    x_to_anchor = D_anch.min(dim=-1).indices
    anchor_to_x = D_anch.mT.topk(dim = -1, k = anchor_K, largest = False).indices
    x_anchor_to_x_inds = torch.gather(anchor_to_x, dim = 1, index = x_to_anchor[...,None].expand(B,N,anchor_K))
    approx_neighbours = torch.gather(x, dim = 1, index = x_anchor_to_x_inds[...,None].expand(B,N,anchor_K,D).reshape(B, N*anchor_K, D)).reshape(B, N, anchor_K, D)
    D_approx = (x[:,:,None] - approx_neighbours).norm(dim=-1)
    approx_knn_inds = D_approx.topk(dim = -1, k = K, largest = False).indices
    approx_neighbours = torch.gather(approx_neighbours, dim = -2, index = approx_knn_inds[...,None].expand(B,N,K,D).reshape(B, N, K, D))
    return approx_neighbours


@torch.no_grad()
def compute_exact_knn(x, K): 
    # x = (B,N,D)
    # computation time: N * num_anchors + N * anchor_K
    # memory: same kind of
    # Should not be run for N < 1000
    # Also requires too much memory if D >> 1
    B,N,D = x.shape
    Dist = torch.cdist(x,x)
    exact_knn_inds = Dist.topk(dim = -1, k = K, largest = False).indices
    exact_neighbours = torch.gather(x, dim = -2, index = exact_knn_inds[...,None].expand(B,N,K,D).reshape(B, N*K, D)).reshape(B, N, K, D)
    return exact_neighbours

def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_

@torch.no_grad()
def extract_matches(all_scores, all_matchability = None, score_threshold = -3, mask_A = None, mask_B = None, mode = "final"):
    if mode == "avg":
        avg_score = sum(all_scores) / len(all_scores)
        scores = avg_score
    elif mode == "final":
        scores = all_scores[-1]
    if all_matchability is not None:
        matchability_A = all_matchability[-1][0]
        matchability_B = all_matchability[-1][1]
        scores = scores + matchability_A[...,None] + matchability_B[...,None,:]
    top1_score = scores.flatten().max()
    score_threshold = min(top1_score, score_threshold)
    matches = torch.nonzero((scores == scores.max(dim=-1, keepdim = True).values) * (scores == scores.max(dim=-2, keepdim = True).values) * (scores >= score_threshold), as_tuple=True)
    confidence = F.sigmoid(scores[matches[0], matches[1], matches[2]])
    return matches, confidence

def collate_ragged(batch):
    collated_batch = {}
    collated_batch["is_ragged"] = True
    collated_batch["batch_size"] = len(batch)
    for sample in batch:
        for k,v in sample.items():
            if isinstance(v, torch.Tensor):
                collated_batch[k] = torch.cat((collated_batch[k],v)) if k in collated_batch else v
            else:
                collated_batch[k] = collated_batch[k] + [v] if k in collated_batch else [v]
    return collated_batch


# From lightglue
def pad_to_length(x: torch.Tensor, length: int):
    if length <= x.shape[-2]:
        return x, torch.ones_like(x[..., :1], dtype=torch.bool)
    pad = torch.ones(*x.shape[:-2], length-x.shape[-2], x.shape[-1],
                     device=x.device, dtype=x.dtype)
    y = torch.cat([x, pad], dim=-2)
    mask = torch.zeros(*y.shape[:-1], 1, dtype=torch.bool, device=x.device)
    mask[..., :x.shape[-2], :] = True
    return y, mask

def pad_to_length_ragged(x: torch.Tensor, lengths, pad_length: int):
    x_pads, masks = [], []
    cum_length = torch.cumsum(torch.cat((torch.zeros_like(lengths[:1]),lengths)), 0)
    for idx in range(len(cum_length)-1):
        x_pad, mask = pad_to_length(x[cum_length[idx]:cum_length[idx+1]], pad_length)
        x_pads.append(x_pad)
        masks.append(mask)
    return torch.stack(x_pads), torch.stack(masks)

def estimate_normals(x, view_point = None):
    is_tensor = isinstance(x, torch.Tensor) 
    if is_tensor:
        device = x.device
        x = x.cpu().numpy()
    o3d_x = to_o3d_pcd(x)
    if view_point is None:
        view_point = np.zeros(3)
    o3d_x.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
    normals = np.asarray(o3d_x.normals)
    normals = normal_redirect(x, normals, view_point = view_point)
    if is_tensor:
        normals = torch.from_numpy(normals).to(device)
    return normals

def random_cut_pointclouds(pointcloud_A, pointcloud_B, min_overlap = 0.3):
    rand_plane_normal = np.random.randn(3)
    rand_plane_normal = rand_plane_normal / np.linalg.norm(rand_plane_normal)
    pivot_point = random.choice(pointcloud_A)
    on_positive_side_A = (pointcloud_A - pivot_point[None,:]) @ rand_plane_normal > 0
    if on_positive_side_A.mean() < 0.5:
        on_positive_side_A = ~on_positive_side_A
        rand_plane_normal = -rand_plane_normal
    overlap_B = (pointcloud_B - pivot_point[None,:]) @ rand_plane_normal > 0
    
    rand_plane_normal = np.random.randn(3)
    rand_plane_normal = rand_plane_normal / np.linalg.norm(rand_plane_normal)
    pivot_point = random.choice(pointcloud_B)
    on_positive_side_B = (pointcloud_B - pivot_point[None,:]) @ rand_plane_normal > 0
    if on_positive_side_B.mean() < 0.5:
        on_positive_side_B = ~on_positive_side_B
        rand_plane_normal = -rand_plane_normal
    overlap_A = (pointcloud_A - pivot_point[None,:]) @ rand_plane_normal > 0
    
    enough_overlap_A = ((on_positive_side_A * overlap_A).sum() / (on_positive_side_A.sum()+1)) > min_overlap and (overlap_A.mean() > 0.5)
    enough_overlap_B = ((on_positive_side_B * overlap_B).sum() / (on_positive_side_B.sum()+1)) > min_overlap and (overlap_B.mean() > 0.5)
    
    if enough_overlap_A and enough_overlap_B:
        return on_positive_side_A, on_positive_side_B
    else:
        return random_cut_pointclouds(pointcloud_A, pointcloud_B, min_overlap = min_overlap)


def transform(pts, transform):
    pts = (transform[:3,:3]@pts.T).T + transform[:3,3]
    return pts

def extract_pcd_from_colmap_model(model):
    pt = np.array([pt.xyz for pt in model.points3D.values()])
    viewpoints = np.array([model.images[pt.track.elements[0].image_id].projection_center() for pt in model.points3D.values()])
    rgb = np.array([pt.color for pt in model.points3D.values()])
    return pt, viewpoints, rgb