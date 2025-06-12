import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from easydict import EasyDict as edict

from colabsfm.utils import load_obj, get_best_device
from colabsfm.models.model import OmniGlue
from colabsfm.train import train_epoch, train_k_steps
from colabsfm.datasets.indoor import IndoorDataset
from colabsfm.losses.nll_loss import NLLLoss
from colabsfm.benchmarks.tdmatch import TDMatchBenchmark
from colabsfm.checkpoint import CheckPoint
import colabsfm
from tensorboardX import SummaryWriter

def train(args):
    import os
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    colabsfm.LOGGER = SummaryWriter(logdir = os.path.join(args.log_dir, experiment_name))
    
    config = edict(normalize_descriptors = False, backbone = 'roitr', root=f"{args.data_root}/indoor", colabsfm_root=f"{args.data_root}/colabsfm/0015",
                   overlap_radius = 0.0375, augment_noise = 0.005, encoder_only = False, n_layers = 8,
                   use_global_transformer = True, nsample = [8, 16, 16, 16], use_refiner = True, normalize_features = False,
                   refiner_num_heads = 1, refine_level = 2, refiner_n_layers = 8, refiner_descriptor_dim = 128, refine_num_nn = 32, refiner = "local_softmax",
                   ppf_mode = "no_dists",
                   signature = experiment_name, return_dict = True)
    model = OmniGlue(
        config,
        ).to(get_best_device())
    #model = nn.DataParallel(model)
    dataset = IndoorDataset(data_root= args.data_root, infos = load_obj(f"{args.data_root}/indoor/train_info.pkl"), config = config, max_points=8_000)# TODO: more max points
    tdmatch_benchmark = TDMatchBenchmark(IndoorDataset(data_root= args.data_root, infos = load_obj(f"{args.data_root}/indoor/val_info.pkl"), config = config))
    batch_size = 2
    from colabsfm.utils import collate_ragged
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers = batch_size, shuffle=True, collate_fn = collate_ragged)
    optimizer = AdamW(model.parameters(), lr = batch_size * 4e-5, weight_decay=1e-8)
    lr_scheduler = ConstantLR(factor = 1, optimizer=optimizer)
    objective = NLLLoss()
    num_epochs = 150
    num_gpus = 1
    effective_batch_size = batch_size * num_gpus
    num_steps = len(dataloader) * effective_batch_size * num_epochs
    checkpointer = CheckPoint(dir = args.checkpoint_dir, name = experiment_name)
    start_step = 0
    model, optimizer, lr_scheduler, start_step = checkpointer.load(model, optimizer, lr_scheduler, start_step)
    if args.colabsfm_benchmark and False:
        from colabsfm.datasets.colabsfm import MegaDepth
        import json
        colabsfm_benchmark = TDMatchBenchmark(MegaDepth(data_root= "data/colabsfm", infos = json.load(open("data/colabsfm/0015/disk+lightglue_infos.json","r")), config = config))
        colabsfm_benchmark.benchmark(model)
    #tdmatch_benchmark.benchmark(model)
    colabsfm.GLOBAL_STEP = start_step
    for step in range(start_step, num_steps, len(dataloader) * effective_batch_size):
        train_epoch(
            dataloader = dataloader, 
            model = model, 
            objective = objective, 
            optimizer = optimizer, 
            lr_scheduler = lr_scheduler,
            iters_to_accumulate = 1,
            )        
        checkpointer.save(model, optimizer, lr_scheduler, colabsfm.GLOBAL_STEP)
        tdmatch_benchmark.benchmark(model)

def test(args):
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", required=False)
    parser.add_argument("--checkpoint_dir", default = "workspace", required=False)
    parser.add_argument("--log_dir", default = "logs", required=False)
    parser.add_argument("--colabsfm_benchmark", default = "logs", action= "store_true")
    args,_ = parser.parse_known_args()
    train(args)
    


