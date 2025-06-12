import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training

from colabsfm.roitr.lib.utils import setup_seed
from colabsfm.configs.utils import load_config
from easydict import EasyDict as edict
from colabsfm.roitr.dataset.dataloader import get_dataset, get_dataloader
from colabsfm.roitr.model.RIGA_v2 import create_model
from colabsfm.roitr.lib.loss import OverallLoss, Evaluator
from colabsfm.roitr.lib.tester import get_trainer
from colabsfm.models import RoITr

def main(points_A, points_B, viewpoints_A = None, viewpoints_B = None, shared_scale = True):
    from tensorboardX import SummaryWriter
    import colabsfm

    #########################################################
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--data_root", type = str, default = "data")
    parser.add_argument("--log_dir", type = str, default = "logs")
    parser.add_argument("--checkpoint_dir", type = str, default = "workspace")


    #"--data_root", data_path, "--checkpoint_dir", checkpoint_path, "--log_dir", log_path
    args, _ = parser.parse_known_args()
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    colabsfm.LOGGER = SummaryWriter(logdir = os.path.join(args.log_dir, experiment_name))
    config = load_config(args.config)
    config['root'] = args.data_root + ("/indoor" if config['dataset'] == "tdmatch" else "/colabsfm/pointclouds")
    config['local_rank'] = int(os.environ.get("LOCAL_RANK", -1))
    #########################################################
    #set cuda devices for both DDP training and single-GPU training
    if config['local_rank'] > -1:
        torch.cuda.set_device(config['local_rank'])
        config['device'] = torch.device('cuda', config['local_rank'])
        torch.distributed.init_process_group(backend='nccl')

    else:
        torch.cuda.set_device(0)
        config['device'] = torch.device('cuda', 0)

    ##########################################################
    setup_seed(42) # fix the seed

    ##########################################################
    # set paths for storing results
    config['snapshot_dir'] = f"{args.log_dir}/{experiment_name}"#"/tmp"#'{}/{}'.format(args.data_root, config['exp_dir'])
    config['tboard_dir'] = f"{args.log_dir}/{experiment_name}"#'{}/{}/tensorboard'.format(args.data_root, config['exp_dir'])
    config['save_dir'] = f"{args.checkpoint_dir}/{experiment_name}"#'{}/{}/checkpoints'.format(args.data_root, config['exp_dir'])
    os.makedirs(config['save_dir'], exist_ok = True)
    config['visual_dir'] = args.log_dir#'{}/{}/visualization'.format(args.data_root, config['exp_dir'])
    ##########################################################
    config = edict(config)
    assert config.mode in ("val", "test"), "no training"
    ######## Experiment settings ###########
    experiment_conf = edict(
        lr = 4e-5,
        use_glue = False,
        glue_refine = False,
    )
    config.update(experiment_conf)
    config.pretrain = "pretrained/model_3dmatch.pth"

    ##################################################################
    # create model
    model = create_model(config).to(config.device)
    state = torch.load(config.pretrain)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model = RoITr(model).eval()


    results = model.register(torch.from_numpy(points_A).cuda(), torch.from_numpy(points_B).cuda(), 
                             viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B, shared_scale = shared_scale,
                             )
    #print(results)
    #np.save("transform",results['transformation'])
    #print(results['transformation'])
    return results

if __name__ == '__main__':
    #os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp"
    os.environ["OMP_NUM_THREADS"] = "16"
    import numpy as np
    method = "sift-sattler"#"sift-sattler" #"sfm_disk+lightglue" #
    scene = "StMarysChurch"
    points_A = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/train/{method}_cloud.npy")
    viewpoints_A = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/train/{method}_viewpoints.npy")
    points_B = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/test/{method}_cloud.npy")
    viewpoints_B = np.load(f"data/colabsfm/cambridge/{scene}_benchmark/pointclouds/test/{method}_viewpoints.npy")
    import open3d as o3d
    reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

    #scene = "0004"

    #points_A = np.load(f"data/colabsfm/pointclouds_trajectory/{scene}/{method}_cloud_trajectory_0.npy")
    #viewpoints_A = np.load(f"data/colabsfm/pointclouds_trajectory/{scene}/{method}_viewpoints_trajectory_0.npy")
    #points_B = np.load(f"data/colabsfm/pointclouds_trajectory/{scene}/{method}_cloud_trajectory_5.npy")
    #viewpoints_B = np.load(f"data/colabsfm/pointclouds_trajectory/{scene}/{method}_viewpoints_trajectory_5.npy")

    #points_A = 0.4*np.load(f"data/colabsfm/radiotower/CDU/{method}_cloud.npy")
    #viewpoints_A = 0.4*np.load(f"data/colabsfm/radiotower/CDU/{method}_viewpoints.npy")
    #points_B = np.load(f"data/colabsfm/radiotower/TSO/{method}_cloud.npy")
    #viewpoints_B = np.load(f"data/colabsfm/radiotower/TSO/{method}_viewpoints.npy")



    main(points_A, points_B, viewpoints_A = viewpoints_A, viewpoints_B = viewpoints_B, shared_scale = True)
