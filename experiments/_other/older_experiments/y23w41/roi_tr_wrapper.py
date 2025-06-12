import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training
from colabsfm.utils import estimate_normals
import sys
sys.path.append("third_party/RoITr")
from lib.utils import setup_seed
from configs.utils import load_config
from easydict import EasyDict as edict
from dataset.dataloader import get_dataset, get_dataloader
from model.RIGA_v2 import create_model
from lib.loss import OverallLoss, Evaluator
from lib.tester import get_trainer

class RoITrWrapper(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
    @torch.inference_mode()
    def match(self, pointcloud_A, pointcloud_B, normals_A = None, normals_B = None, score_threshold = -3):
        pointcloud_A, pointcloud_B = pointcloud_A.float(), pointcloud_B.float()
        self.train(False)
        if normals_A is None or normals_B is None:
            normals_A = estimate_normals(pointcloud_A).float()
            normals_B = estimate_normals(pointcloud_B).float()
        data = {"pc_A": pointcloud_A, "pc_B": pointcloud_B, 
                "normals_A": normals_A, "normals_B": normals_B,
                "num_pt_A": torch.tensor(pointcloud_A.shape[:1], device = pointcloud_A.device),
                "num_pt_B": torch.tensor(pointcloud_B.shape[:1], device = pointcloud_B.device),
                "is_ragged": True,
                "batch_size": 1,
                }
        data = self(data, score_threshold = score_threshold)
        #Below needed for Predator eval compat
        return data

def main():
    from tensorboardX import SummaryWriter
    import colabsfm
    colabsfm.LOGGER = SummaryWriter(logdir = os.path.join("logs", "roitr"))

    #########################################################
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--local_rank", type=int, default=-1) # for DDP training
    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    config['local_rank'] = args.local_rank
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
    config['snapshot_dir'] = 'snapshot/{}'.format(config['exp_dir'])
    config['tboard_dir'] = 'snapshot/{}/tensorboard'.format(config['exp_dir'])
    config['save_dir'] = 'snapshot/{}/checkpoints'.format(config['exp_dir'])
    config['visual_dir'] = 'snapshot/{}/visualization'.format(config['exp_dir'])
    ##########################################################
    config = edict(config)

    ##################################################################
    # create model
    #config.model = RIGA(config=config).to(config.device)
    model = create_model(config).to(config.device)

    print("hej")
if __name__ == '__main__':
    main()
