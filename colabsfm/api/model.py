import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training

from colabsfm.roitr.lib.utils import setup_seed
from colabsfm.roitr.configs.utils import load_config
from easydict import EasyDict as edict
from colabsfm.roitr.dataset.dataloader import get_dataset, get_dataloader
from colabsfm.roitr.model.RIGA_v2 import create_model
from colabsfm.roitr.lib.loss import OverallLoss, Evaluator
from colabsfm.roitr.lib.tester import get_trainer
from colabsfm.models import SimTr

def sfmregger(weights_path):
    from tensorboardX import SummaryWriter
    import colabsfm

    #########################################################
    # load config
    parser = argparse.ArgumentParser()

    #"--data_root", data_path, "--checkpoint_dir", checkpoint_path, "--log_dir", log_path
    args, _ = parser.parse_known_args()
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    config = load_config(args.config)
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
    config = edict(config)
    assert config.mode in ("val", "test"), "no training"
    ######## Experiment settings ###########
    config.pretrain = weights_path
    
    experiment_conf = edict(
        name = 'omniglue',  # just for interfacing
        descriptor_dim = 256,
        lr = 16e-5,
        refiner_num_heads = 1, 
        refiner_n_layers = 4, 
        refiner_descriptor_dim = 64,
        glue_detach = True,
        glue_refine = True,
        flash = True,
        fine_loss_use_mnn = False,
        iter_size = 4,
    )
    config.update(experiment_conf)


    ##################################################################
    # create model
    model = create_model(config).to(config.device)
    state = torch.load(config.pretrain)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model = SimTr(model).eval()
    return model