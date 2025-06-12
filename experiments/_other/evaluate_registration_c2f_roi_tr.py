import os, sys, glob, torch, argparse
import numpy as np
from colabsfm.roitr.lib.utils import setup_seed, natural_key
from tqdm import tqdm
from colabsfm.roitr.registration.benchmark_utils import ransac_pose_estimation_correspondences, get_inlier_ratio_correspondence, get_scene_split, write_est_trajectory
from colabsfm.roitr.registration.benchmark import benchmark
from colabsfm.roitr.lib.utils import square_distance
from colabsfm.roitr.visualizer.visualizer import Visualizer
from colabsfm.roitr.visualizer.plot import draw_distance_geo_feat
from colabsfm.roitr.dataset.common import collect_local_neighbors, get_square_distance_matrix, point2node_sampling
from colabsfm.roitr.lib.utils import weighted_procrustes

setup_seed(0)


def extract_correspondence(dist, major='row'):
    if major == 'row':
        top2 = np.partition(dist, axis=1, kth=1)[:, :2]
        row_inds = np.arange(dist.shape[0])
        d0 = top2[:, 0]
        d1 = top2[:, 1]
        col_inds = np.argmin(dist, axis=1)
        nn = np.where(d0 < d1, d0, d1)
        nn2 = np.where(d0 > d1, d0, d1)
        weights = -nn
    elif major == 'col':
        top2 = np.partition(dist, axis=0, kth=1)[:2, :]
        col_inds = np.arange(dist.shape[1])
        d0 = top2[0, :]
        d1 = top2[1, :]
        row_inds = np.argmin(dist, axis=0)
        nn = np.where(d0 < d1, d0, d1)
        nn2 = np.where(d0 > d1, d0, d1)
        weights = -nn
    else:
        raise NotImplementedError

    return row_inds, col_inds, weights



def benchmark_registration(desc, exp_dir, whichbenchmark, n_points, ransac_with_mutual=False, inlier_ratio_threshold=0.05):
    gt_folder = f'third_party/configs/benchmarks/{whichbenchmark}'
    exp_dir = f'{exp_dir}/{whichbenchmark}/{n_points}'
    if (not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)

    results = dict()
    results['w_mutual'] = {'inlier_ratios': [], 'distances': []}
    results['wo_mutual'] = {'inlier_ratios': [], 'distances': []}
    tsfm_est = []
    inlier_ratio_list = []

    coarse_sample = 256
    idx = 0
    for eachfile in tqdm(sorted(desc)):

        #if idx < 1320:
        #    idx += 1
        #    continue
        #else:
        #    idx += 1
        ######################################################
        # 1. take the nodes and descriptors
        print(eachfile)
        data = torch.load(eachfile)
        src_pcd, tgt_pcd = data['src_pcd'], data['tgt_pcd']
        rot, trans = data['rot'], data['trans']
        src_corr_pts, tgt_corr_pts = data['src_corr_pts'], data['tgt_corr_pts']
        confidence = data['confidence']
        ######################################################
        # 2. run ransac
        prob = confidence / torch.sum(confidence)
        print(confidence.shape[0])
        if prob.shape[0] > n_points:
            sel_idx = np.random.choice(prob.shape[0], n_points, replace=False, p=prob.numpy())
            #mute the previous line and unmute the following line for changing the sampling strategy to top-k
            #sel_idx = torch.topk(confidence, k=n_points)[1]
            src_corr_pts, tgt_corr_pts = src_corr_pts[sel_idx], tgt_corr_pts[sel_idx]
            confidence = confidence[sel_idx]

        correspondences = torch.from_numpy(np.arange(src_corr_pts.shape[0])[:, np.newaxis]).expand(-1, 2)
        tsfm_est.append(ransac_pose_estimation_correspondences(src_corr_pts, tgt_corr_pts, correspondences))
        
        gt_transformed_pcd = torch.matmul(src_pcd, rot.T) + trans.T
        est_rot, est_trans = tsfm_est[-1][:3, :3], tsfm_est[-1][:3, -1:]
        est_transformed_pcd = torch.matmul(src_pcd, torch.from_numpy(est_rot.T).float()) + torch.from_numpy(est_trans.T).float()
        cur_RMSE = torch.mean(torch.sqrt(torch.sum((est_transformed_pcd - gt_transformed_pcd) ** 2, dim=-1)))
        print(cur_RMSE)
        ######################################################
        # 3. calculate inlier ratios
        if whichbenchmark == "colabsfm":
            scale = tgt_pcd.std().item()
        else:
            scale = 1
        cur_inlier_ratio = get_inlier_ratio_correspondence(src_corr_pts, tgt_corr_pts, rot, trans, inlier_distance_threshold=scale*0.1)
        inlier_ratio_list.append(cur_inlier_ratio)
        idx += 1

    # tsfm_est = np.array(tsfm_est)

    ########################################
    # wirte the estimated trajectories
    # write_est_trajectory(gt_folder, exp_dir, tsfm_est)

    ########################################
    # evaluate the results, here FMR and Inlier ratios are all average twice
    inlier_ratio_list = np.array(inlier_ratio_list)
    #benchmark(exp_dir, gt_folder)
    split = get_scene_split(whichbenchmark)

    inliers = []
    fmrs = []
    inlier_ratio_thres = 0.05
    for ele in split:
        c_inliers = inlier_ratio_list[ele[0]:ele[1]]
        inliers.append(np.mean(c_inliers))
        fmrs.append((np.array(c_inliers) > inlier_ratio_thres).mean())
    with open(os.path.join(exp_dir, 'result'), 'a') as f:
        f.write(f'Inlier ratio: {np.mean(inliers):.3f} : +- {np.std(inliers):.3f}\n')
        f.write(f'Feature match recall: {np.mean(fmrs):.3f} : +- {np.std(fmrs):.3f}\n')

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', default="logs/y23w42_refine_glue_4e-5_refine_mnn/3DLoMatch", type=str, help='path to precomputed features and scores') # logs/y23w42_refine_glue_4e-5_refine_mnn/3DLoMatch # data/tdmatch_ripoint_transformer_test/3DMatch
    parser.add_argument('--benchmark', default='3DLoMatch', type=str, help='[3DMatch, 3DLoMatch, colabsfm]')
    parser.add_argument('--n_points', default=1000, type=int, help='number of points used by RANSAC')
    parser.add_argument('--exp_dir', default='est_traj', type=str, help='export final results')
    args, _ = parser.parse_known_args()
    desc = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)
    benchmark_registration(desc, args.exp_dir, args.benchmark, args.n_points, ransac_with_mutual=False)

