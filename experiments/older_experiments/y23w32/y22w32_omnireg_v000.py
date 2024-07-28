import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from types import SimpleNamespace

from colabsfm.utils import load_obj, get_best_device
from colabsfm.models.model import OmniGlue
from colabsfm.train import train_k_epochs
from colabsfm.datasets.indoor import IndoorDataset
from colabsfm.losses.nll_loss import loss


def train(args):
    model = OmniGlue().to(get_best_device())
    #model = nn.DataParallel(model)
    config = dict(root="data/indoor", overlap_radius = 0.0375, augment_noise = 0.005)
    config = SimpleNamespace(**config)
    dataset = IndoorDataset(infos = load_obj("data/indoor/train_info.pkl"), config = config)
    dataloader = DataLoader(dataset, batch_size=2, num_workers = 6)
    optimizer = AdamW(model.parameters(), lr = 1e-4, weight_decay=1e-8)
    lr_scheduler = ConstantLR(optimizer=optimizer)
    objective = loss
    train_k_epochs(
        0,
        100,
        dataloader=dataloader, 
        model = model, 
        objective = objective, 
        optimizer = optimizer, 
        lr_scheduler = lr_scheduler,
        iters_to_accumulate=1,
        )

def test(args):
    pass


if __name__ == "__main__":
    train(None)
    


