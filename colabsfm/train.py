import torch
from tqdm import tqdm
from colabsfm.utils import to_best_device
import colabsfm
from time import perf_counter

def train_step(data, model, objective, optimizer, grad_scaler = None, do_opt_step = True, **kwargs):
    t0 = perf_counter()
    
    data = model(data)
    l = objective(data)
    if grad_scaler is not None:
        grad_scaler.scale(l).backward()
        if do_opt_step:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
    else:
        l.backward()
        t3 = perf_counter()
        if do_opt_step:
            optimizer.step()
            optimizer.zero_grad()
        t4 = perf_counter()
    colabsfm.GLOBAL_STEP = colabsfm.GLOBAL_STEP + data["batch_size"]
    return {"train_out": data, "train_loss": l.item()}


def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler = None, progress_bar=True, iters_to_accumulate = 1,
):
    for n in tqdm(range(n_0, n_0 + k), disable=not progress_bar, mininterval = 10.):
        data = next(dataloader)
        model.train(True)
        data = to_best_device(data)
        do_opt_step = (n % iters_to_accumulate) == 0
        train_step(
            data=data,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n=n,
            grad_scaler = grad_scaler,
            do_opt_step=do_opt_step,
        )
        lr_scheduler.step()


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
    iters_to_accumulate = 1
):
    model.train(True)
    print(f"At epoch {epoch}")
    for idx, data in tqdm(enumerate(dataloader), mininterval=5.0):
        data = to_best_device(data)
        t0 = perf_counter()
        do_opt_step = (idx % iters_to_accumulate) == 0
        train_step(
            data=data, 
            model=model, 
            objective=objective, 
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            do_opt_step = do_opt_step,
        )
        t1 = perf_counter()
        train_step_time = t1 - t0 
        #print(f"{train_step_time=}")
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler, iters_to_accumulate = 1
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            iters_to_accumulate=iters_to_accumulate,
        )