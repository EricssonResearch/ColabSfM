import argparse
import torch

from colabsfm.roitr.lib.utils import setup_seed
from colabsfm.configs.utils import load_config
from easydict import EasyDict as edict
from colabsfm.roitr.model.RIGA_v2 import create_model
from colabsfm.models import RoITr
from urllib.parse import urlparse

def _is_url(path):
    """Check if path is a URL"""
    try:
        result = urlparse(str(path))
        return all([result.scheme, result.netloc])
    except:
        return False

def load_RoITr(weights_path):

    #########################################################
    # load config
    parser = argparse.ArgumentParser()

    args, _ = parser.parse_known_args()
    if 'config' in args:
        config = load_config(args.config)
    else:
        config = load_config("colabsfm/configs/val/megadepth.yaml")
    # if config['local_rank'] > -1:
    #     torch.cuda.set_device(config['local_rank'])
    #     config['device'] = torch.device('cuda', config['local_rank'])
    #     torch.distributed.init_process_group(backend='nccl')

    # else:
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
    if _is_url(config.pretrain):
        state = torch.hub.load_state_dict_from_url(
            url=config.pretrain,
            map_location=config.device,
            progress=True
        )
    else:
        state = torch.load(config.pretrain)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
    model = RoITr(model).eval()
    return model