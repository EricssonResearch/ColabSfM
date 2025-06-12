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
from colabsfm.roitr.lib.loss import OverallLoss, Evaluator, EvaluatorRegistration
from colabsfm.roitr.lib.tester import get_trainer
from colabsfm.models.geotr_wrapper import GeoTr, calibrate_neighbors_stack_mode
from colabsfm.geotransformer import create_model
from colabsfm.geotransformer.config.config import make_cfg

from colabsfm.geotransformer.utils.data import registration_collate_fn_stack_mode
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
    parser.add_argument("--output_metric_path", type=str, default = None)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--sim3", action='store_true')

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
    if args.sim3:
        experiment_conf = edict(
            colabsfm_mode = "sim3",
        )
        config.update(experiment_conf)
    

    if args.backbone == "3dmatch":
        config.pretrain = "pretrained/geotransformer-3dmatch.pth.tar"
        backbone = "3dmatch"
        cfg = make_cfg()

        cfg.backbone.init_voxel_size = 0.05
        # cfg.backbone.base_radius  = 2.25
        neighs = None
    else:
        backbone = "kitti"
        config.pretrain = "pretrained/geotransformer-kitti.pth.tar"
        # from colabsfm.geotransformer.config.kitti_config import make_cfg
        cfg = make_cfg()
        neighs = [411,  73,  64,  45,  27]
        
    cfg.backbone.init_radius = cfg.backbone.init_voxel_size * cfg.backbone.base_radius
    cfg.test.point_limit = 30000
    cfg.debug = args.debug

    # create dataset and dataloader
    train_set, val_set, benchmark_set = get_dataset(config)
    if config.mode == 'train': set_ = train_set
    elif config.mode == 'val': set_ = val_set
    elif config.mode == 'test': set_ = benchmark_set

    print("Computing neighbors")
    config.num_stages  = cfg.backbone.num_stages
    config.init_voxel_size  = cfg.backbone.init_voxel_size
    config.base_radius  = cfg.backbone.base_radius
    config.init_radius = cfg.backbone.init_radius
    if neighs is None:
        config.neighbors =  calibrate_neighbors_stack_mode(
            set_,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius, 
        )
    else:
        config.neighbors = neighs
    print(config.neighbors)

    ##################################################################
    # create model
    config.model = GeoTr(create_model(cfg, backbone).to(config.device), config)

    # print the details of network architecture
    # if config.local_rank <= 0:
    #     print(config.model)
    # for PyTorch DistbutedDataParallel(DDP) training
    if config.local_rank >= 0:
        config.model = torch.nn.parallel.DistributedDataParallel(config.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)


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
    config.evaluator = EvaluatorRegistration(config)
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
    main()
