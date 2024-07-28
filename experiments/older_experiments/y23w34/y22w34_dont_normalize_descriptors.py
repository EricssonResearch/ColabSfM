import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from types import SimpleNamespace

from colabsfm.utils import load_obj, get_best_device
from colabsfm.models.model import OmniGlue
from colabsfm.train import train_epoch, train_k_steps
from colabsfm.datasets.indoor import IndoorDataset
from colabsfm.losses.nll_loss import NLLLoss
from colabsfm.benchmarks.tdmatch import TDMatchBenchmark
from colabsfm.checkpoint import CheckPoint

def train(args):
    import os
    experiment_name = os.path.splitext(os.path.basename(__file__))[0]
    normalize_descriptors = False
    model = OmniGlue(
        normalize_descriptors=normalize_descriptors).to(get_best_device())
    #model = nn.DataParallel(model)
    config = dict(root="data/indoor", overlap_radius = 0.0375, augment_noise = 0.005)
    config = SimpleNamespace(**config)
    dataset = IndoorDataset(infos = load_obj("data/indoor/train_info.pkl"), config = config)
    tdmatch_benchmark = TDMatchBenchmark(IndoorDataset(infos = load_obj("data/indoor/val_info.pkl"), config = config))
    dataloader = DataLoader(dataset, batch_size=2, num_workers = 0, shuffle=True)
    optimizer = AdamW(model.parameters(), lr = 8e-5, weight_decay=1e-8)
    lr_scheduler = ConstantLR(optimizer=optimizer)
    objective = NLLLoss()
    num_epochs = 1000
    checkpointer = CheckPoint(dir = "workspace", name = experiment_name)
    start_epoch = 0
    model, optimizer, lr_scheduler, start_epoch = checkpointer.load(model, optimizer, lr_scheduler, start_epoch)
    #tdmatch_benchmark.benchmark(model)
    for epoch_num in range(start_epoch, num_epochs):
        train_epoch(
            dataloader = dataloader, 
            model = model, 
            objective = objective, 
            optimizer = optimizer, 
            lr_scheduler = lr_scheduler,
            iters_to_accumulate = 1,
            )        
        checkpointer.save(model, optimizer, lr_scheduler, epoch_num)
        tdmatch_benchmark.benchmark(model)

def test(args):
    pass


if __name__ == "__main__":
    train(None)
    


