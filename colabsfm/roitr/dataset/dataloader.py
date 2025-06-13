import torch
from functools import partial
from .common import load_info, collate_fn
from .tdmatch import TDMatchDataset
from .fdmatch import FDMatch
from colabsfm.datasets.colabsfm import ColabSfM
import multiprocessing

def get_dataset(config):
    '''
    Make pytorch dataset for training, validation and testing
    :param config: configuration
    :return: train_set: training dataset
             val_set: validation dataset
             benchmark_set: testing dataset
    '''
    if config.dataset == 'tdmatch':
        info_train = load_info(config.train_info)
        info_val = load_info(config.val_info)
        info_benchmark = load_info(f'colabsfm/configs/tdmatch/{config.benchmark}.pkl')
        training_set = TDMatchDataset(info_train, config, data_augmentation=True)
        val_set = TDMatchDataset(info_val, config, data_augmentation=False)
        testing_set = TDMatchDataset(info_benchmark, config, data_augmentation=False)
    elif config.dataset == 'fdmatch':
        training_set = FDMatch(config, 'train', data_augmentation=True)
        val_set = FDMatch(config, 'val', data_augmentation=False)
        testing_set = FDMatch(config, 'test', data_augmentation=False)
    elif config.dataset in ['megadepth','quad']:
        info_train = load_info(config.train_info)
        info_val = load_info(config.val_info)
        info_benchmark = load_info(config.test_info)

        training_set = ColabSfM(info_train, config, 'train', data_augmentation=True)
        val_set = ColabSfM(info_val, config, 'val', data_augmentation=False)
        testing_set = ColabSfM(info_benchmark, config, 'test', data_augmentation=False)

    else:
        raise NotImplementedError

    return training_set, val_set, testing_set


def get_dataloader(dataset, sampler=None, batch_size=1, num_workers=8, shuffle=True, drop_last=True):
    '''
    Get the pytorch dataloader for specific pytorch dataset
    :param dataset: pytorch dataset
    :param batch_size: size of a batch of data
    :param num_workers: the number of threads used in dataloader
    :param shuffle: whether to shuffle dataset for each epoch
    :return: pytorch dataloader
    '''
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=partial(collate_fn, config=dataset.config),
        drop_last=drop_last,
        multiprocessing_context=multiprocessing.get_context("spawn"),
    )
    return data_loader

