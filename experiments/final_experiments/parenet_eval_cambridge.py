import copy
import os, argparse, json, shutil
import torch
from torch import optim
import numpy as np
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training

from colabsfm.roitr.lib.utils import setup_seed
from colabsfm.roitr.configs.utils import load_config
from easydict import EasyDict as edict
from colabsfm.roitr.dataset.dataloader import get_dataset, get_dataloader
from colabsfm.roitr.lib.loss import OverallLoss, Evaluator
from colabsfm.roitr.lib.tester import get_trainer
from colabsfm.models.parenet_wrapper import PareNet#, calibrate_neighbors_cambridge
from colabsfm.pareconv import create_model, make_cfg

import warnings
warnings.filterwarnings("ignore")


def main():
    from tensorboardX import SummaryWriter
    import colabsfm

    #########################################################
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--local_rank", type=int, default=-1) # for DDP training
    parser.add_argument("--data_root", type = str, default = "data/sfmreg/megadepth/pointclouds_megadepth")
    parser.add_argument("--log_dir", type = str, default = "logs")
    parser.add_argument("--checkpoint_dir", type = str, default = "workspace")
    parser.add_argument("--backbone", type = str, default = "3dmatch")
    parser.add_argument("--output_metric_path", type=str, default = None)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--points", type=int, default = 30000)


    #"--data_root", data_path, "--checkpoint_dir", checkpoint_path, "--log_dir", log_path
    args, _ = parser.parse_known_args()
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    colabsfm.LOGGER = SummaryWriter(logdir = os.path.join(args.log_dir, experiment_name))
    config = load_config(args.config)
    config['local_rank'] = args.local_rank
    config['root'] = args.data_root
    #########################################################
    #set cuda devices for both DDP training and single-GPU training
    if config['local_rank'] > -1:
        torch.cuda.set_device(config['local_rank'])
        config['device'] = torch.device('cuda', config['local_rank'])
        torch.distributed.init_process_group(backend='nccl')

    else:
        torch.cuda.set_device(0)
        config['device'] = torch.device('cuda', 0)
    config["output_metric_path"] = args.output_metric_path
    config["debug"] = args.debug


    ##########################################################
    setup_seed(42) # fix the seed

    ##########################################################
    # set paths for storing results
    config['snapshot_dir'] = f"{args.log_dir}/{experiment_name}"#"/tmp"#'{}/{}'.format(args.data_root, config['exp_dir'])
    config['tboard_dir'] = f"{args.log_dir}/{experiment_name}"#'{}/{}/tensorboard'.format(args.data_root, config['exp_dir'])
    config['save_dir'] = args.checkpoint_dir#'{}/{}/checkpoints'.format(args.data_root, config['exp_dir'])
    config['visual_dir'] = args.log_dir#'{}/{}/visualization'.format(args.data_root, config['exp_dir'])
    ##########################################################
    config = edict(config)
    
    ######## Experiment settings ###########
    experiment_conf = edict(
    )
    config.update(experiment_conf)

    neighs = None
    if args.backbone == "3dmatch":
        config.pretrain = "pretrained/parenet-3dmatch.pth.tar"
        backbone = "3dmatch"
        from colabsfm.pareconv import create_model, make_cfg
        cfg = make_cfg()
    else:
        raise NotImplementedError

    # cfg.backbone.init_radius = cfg.backbone.init_voxel_size * cfg.backbone.base_radius
    cfg.test.point_limit = cfg.train.point_limit = args.points
    # cfg.debug = args.debug

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_dataset(config)
    if config.mode == 'train': set_ = train_set
    elif config.mode == 'val': set_ = val_set
    elif config.mode == 'test': set_ = benchmark_set

    method = "sift-sattler"#"sfm_disk+lightglue"#"sp+sg"
    scenes =["GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
    print("Computing neighborhoods")
    config.num_stages  = cfg.backbone.num_stages
    config.init_voxel_size  = cfg.backbone.init_voxel_size
    config.neighbors = cfg.backbone.num_neighbors
    config.subs_ratio = cfg.backbone.subsample_ratio
    config.num_points = cfg.test.point_limit
    neighs = None
    # if neighs is None:
        # init_radius = config.init_voxel_size* 2.25
    #     config.neighbors =  calibrate_neighbors_cambridge(
    #         method,
    #         cfg.backbone.num_stages,
    #         config.init_voxel_size,
    #         init_radius, 
    #         config.subs_ratio,
    #     ) 
    # else:
    config.neighbors = cfg.backbone.num_neighbors
    print(config.neighbors)

    ##################################################################
    # create model
    config.model = PareNet(create_model(cfg).to(config.device), config)
    config.model.eval()
    torch.set_grad_enabled(False)
    state_dict = torch.load(config.pretrain)
    config.model.load_state_dict(state_dict["model"])
    print("Loaded")

    print("Doing cambridge benchmark with SE(3) (shared_scale = True), set shared_scale = False if you want Sim3 benchmark")
    for scene in scenes:
        points_A = np.load(f"{args.data_root}/sfmreg/cambridge/{scene}_benchmark/pointclouds/train/{method}_cloud.npy")
        viewpoints_A = np.load(f"{args.data_root}/sfmreg/cambridge/{scene}_benchmark/pointclouds/train/{method}_viewpoints.npy")
        rot_ab = np.eye(3)
        tr_ab = np.zeros([3])
        if False:
            print("Sampling Random Rotation, Note: For rotation variant methods this needs to be repeated many times to get a reliable estimate")
            from scipy.spatial.transform import Rotation
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            points_A = np.ascontiguousarray((np.matmul(rot_ab,points_A.T).T))
            viewpoints_A = np.ascontiguousarray((np.matmul(rot_ab,viewpoints_A.T).T))
        points_B = np.load(f"{args.data_root}/sfmreg/cambridge/{scene}_benchmark/pointclouds/test/{method}_cloud.npy")
        viewpoints_B = np.load(f"{args.data_root}/sfmreg/cambridge/{scene}_benchmark/pointclouds/test/{method}_viewpoints.npy")
        gt = {"trans": torch.from_numpy(tr_ab.T[None]).float(), 
            "rot": torch.from_numpy(rot_ab.T[None]).float()}
        print("\n\n")
        print(scene, method)

        results = config.model.register(torch.from_numpy(points_A), torch.from_numpy(points_B), 
                                            # viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B, 
                                            shared_scale = True
                                        )
        
        evaluator = Evaluator(config)
        evaluator_stats = evaluator(results, gt)
        print(evaluator_stats)

if __name__ == '__main__':
    main()
