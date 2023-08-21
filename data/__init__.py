"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataloader(opt, is_inference):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt, is_inference)

    phase = 'val' if is_inference else 'training'
    print(f"{phase} dataset [{type(instance).__name__}] of size {len(instance)} was created")

    batch_size = 5 if is_inference else opt.batchSize
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batch_size,
        sampler=data_sampler(instance, shuffle=not is_inference, distributed=True),
        drop_last=not is_inference,
        num_workers= opt.num_workers ,
    )
    return dataloader

def get_train_val_dataloader(opt):
    val_dataset = create_dataloader(opt, is_inference=True)
    train_dataset = create_dataloader(opt, is_inference=False)
    return val_dataset, train_dataset

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)