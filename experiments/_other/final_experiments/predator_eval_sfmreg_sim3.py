import copy
import os, argparse, json, shutil, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training

from colabsfm.roitr.lib.utils import setup_seed
from colabsfm.configs.utils import load_config
from easydict import EasyDict as edict
from colabsfm.roitr.dataset.dataloader import get_dataset, get_dataloader
from colabsfm.roitr.lib.loss import OverallLoss, Evaluator,EvaluatorRegistration
from colabsfm.roitr.lib.tester import get_trainer

from colabsfm.models.pred_wrapper import PreadtorWrap, calibrate_neighbors
from colabsfm.losses.metric_loss_wrapper import MetricLossWraper

sys.path.append("colabsfm/OverlapPredator")
from models.architectures import KPFCNN
from configs.models import architectures



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
    os.makedirs(config['save_dir'], exist_ok = True)
    config['visual_dir'] = args.log_dir#'{}/{}/visualization'.format(args.data_root, config['exp_dir'])
    ##########################################################
    config = edict(config)
    
    ######## Experiment settings ###########
    experiment_conf = edict(
        colabsfm_mode = "sim3",
    )
    config.update(experiment_conf)
    
    if args.backbone == "3dmatch":
        config.pretrain = "pretrained/predator_finetuned_sim3.pth"
        # config.pretrain = '/proj/inter-op-slam/home/ejodhes/checkpoints/y24w7_predator_finetune/model_best_IR.pth'
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

    train_set, val_set, benchmark_set = get_dataset(config)
    if config.mode == 'train': set_ = train_set
    elif config.mode == 'val': set_ = val_set
    elif config.mode == 'test': set_ = benchmark_set

    print("Computing neighborhoods")
    neighbors = [225, 31, 32, 30]#None
    
    config.neighborhood_limits = calibrate_neighbors(set_, config) if neighbors is None else neighbors
    print(config.neighborhood_limits)
    
    
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
    config.loss_func = MetricLossWraper(config)
    config.evaluator = EvaluatorRegistration(config)
    trainer = get_trainer(config)
    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'val':
        trainer.eval()
    elif config.mode == 'test':
        trainer.test()
    else:
        raise NotImplementedError(config.mode)

if __name__ == '__main__':
    main()
