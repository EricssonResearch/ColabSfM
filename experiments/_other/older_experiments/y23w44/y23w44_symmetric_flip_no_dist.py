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


def main():
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

    colabsfm.LOGGER = SummaryWriter(logdir = os.path.join(args.log_dir, experiment_name)) if config.mode == "train" else None 
    ######## Experiment settings ###########
    if config.mode in ("val", "test"):
        config.pretrain = "pretrained/symmetric_flip_no_dist.pth"
    
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
        fine_loss_use_mnn = True,
        iter_size = 4,
        ppf_mode = "flip_to_convex_no_dist",
    )
    config.update(experiment_conf)


    ##################################################################
    # create model
    #config.model = RIGA(config=config).to(config.device)
    config.model = create_model(config).to(config.device)

    # print the details of network architecture
    if config.local_rank <= 0:
        print(config.model)
    # for PyTorch DistbutedDataParallel(DDP) training
    if config.local_rank >= 0:
        config.model = torch.nn.parallel.DistributedDataParallel(config.model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=True)


    # create optimizer
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    # create scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_dataset(config)
    if config.local_rank > -1:
        train_sampler, val_sampler, benchmark_sampler = DistributedSampler(train_set), DistributedSampler(val_set), DistributedSampler(benchmark_set)
    else:
        train_sampler = val_sampler = benchmark_sampler = None

    config.train_loader = get_dataloader(train_set,
                                         sampler=train_sampler,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         shuffle=True,
                                         drop_last=True)
    config.val_loader = get_dataloader(val_set,
                                       sampler=val_sampler,
                                       batch_size=1,
                                       num_workers=config.num_workers,
                                       shuffle=False,
                                       drop_last=False)
    config.test_loader = get_dataloader(benchmark_set,
                                        sampler=benchmark_sampler,
                                        batch_size=1,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        drop_last=False)
    # create losses and evaluation metrics
    config.loss_func = OverallLoss(config)
    config.evaluator = Evaluator(config)
    trainer = get_trainer(config)
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'val':
        trainer.eval()
    elif config.mode == 'test':
        trainer.test()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    #os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp"
    os.environ["OMP_NUM_THREADS"] = "16"
    main()
