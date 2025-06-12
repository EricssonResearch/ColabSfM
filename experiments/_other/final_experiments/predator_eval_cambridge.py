import copy
import os, argparse, json, shutil, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training

from colabsfm.roitr.lib.utils import setup_seed
from colabsfm.configs.utils import load_config
from easydict import EasyDict as edict
from colabsfm.roitr.lib.loss import OverallLoss, Evaluator, EvaluatorRegistration
from colabsfm.roitr.lib.tester import get_trainer

from colabsfm.models.pred_wrapper import PreadtorWrap, calibrate_neighbors_cambridge
import numpy as np
sys.path.append("colabsfm/OverlapPredator")
from models.architectures import KPFCNN
from configs.models import architectures

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
    parser.add_argument("--data_root", type = str, default = "data/colabsfm/megadepth/pointclouds_megadepth")
    parser.add_argument("--log_dir", type = str, default = "logs")
    parser.add_argument("--checkpoint_dir", type = str, default = "workspace")
    parser.add_argument("--backbone", type = str, default = "3dmatch")
    parser.add_argument("--debug", action='store_true')


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
    config["debug"] = True#args.debug
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
    if args.backbone == "3dmatch":
        config.pretrain = "pretrained/predator-3dmatch.pth"
        # config.pretrain = "workspace/y24w7_predator_finetune/model_best_loss.pth"
        # config.pretrain = "pretrained/predator_finetuned.pth"
        architecture = "indoor"
    else:
        raise NotImplementedError
    ##################################################################
    # create model
    config.architecture = architectures[architecture]
    config.model = PreadtorWrap(KPFCNN(config).to(config.device), config)

    # print the details of network architecture
    # if config.local_rank <= 0:
    #     print(config.model)
    # for PyTorch DistbutedDataParallel(DDP) training
    if config.local_rank >= 0:
        config.model = torch.nn.parallel.DistributedDataParallel(config.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    print("Computing neighborhoods")
    neighbors = None
    
    config.neighborhood_limits = calibrate_neighbors_cambridge(config) if neighbors is None else neighbors
    print(config.neighborhood_limits)
    
    method = "sift-sattler"#"sp+sg"#"sift-sattler" #"sfm_disk+lightglue" #
    scenes = ["GreatCourt", "KingsCollege", "OldHospital","ShopFacade", "StMarysChurch"]
    print("Doing cambridge benchmark with SE(3) (shared_scale = True), set shared_scale = False if you want Sim3 benchmark")
    for scene in scenes:
        points_A = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/train/{method}_cloud.npy")
        viewpoints_A = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/train/{method}_viewpoints.npy")
        rot_ab = np.eye(3)
        tr_ab = np.zeros([3])
        if True:
            # print("Sampling Random Rotation, Note: For rotation variant methods this needs to be repeated many times to get a reliable estimate")
            from scipy.spatial.transform import Rotation
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            points_A = np.ascontiguousarray((np.matmul(rot_ab,points_A.T).T))
            viewpoints_A = np.ascontiguousarray((np.matmul(rot_ab,viewpoints_A.T).T))
        points_B = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/test/{method}_cloud.npy")
        viewpoints_B = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/test/{method}_viewpoints.npy")
        gt = {"trans": torch.from_numpy(tr_ab.T[None]).float().cuda(), 
            "rot": torch.from_numpy(rot_ab.T[None]).float().cuda()}
        # print("\n\n")
        print(scene)

        results = config.model.register(torch.from_numpy(points_A).cuda(), torch.from_numpy(points_B).cuda(), 
                                            viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B, shared_scale = True
                                        )
        
        evaluator = EvaluatorRegistration(config)
        evaluator_stats = evaluator(results, gt)
        print(evaluator_stats)

if __name__ == '__main__':
    main()
