"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import numpy as np
from PIL import Image
from io import BytesIO

import random


import json
import util.util as util
import os

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt, is_inference):
        pass

    def get_image_tensor(self, path):
        with self.env.begin(write=False) as txn:
            key = f'{path}'.encode('utf-8')
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        param = get_random_params(img.size, self.scale_param)
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param




def get_random_params(size, scale_param):
    w, h = size
    scale = random.random() * scale_param

    new_w = int( w * (1.0+scale) )
    new_h = int( h * (1.0+scale) )
    x = random.randint(0, np.maximum(0, new_w - w))
    y = random.randint(0, np.maximum(0, new_h - h))
    return {'crop_param': (x, y, w, h), 'scale_size':(new_h, new_w)}


def get_transform(param, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'scale_size' in param and param['scale_size'] is not None:
        osize = param['scale_size']
        transform_list.append(transforms.Resize(osize, interpolation=method))

    if 'crop_param' in param and param['crop_param'] is not None:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, param['crop_param'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __crop(img, pos):
    x1, y1, tw, th = pos
    return img.crop((x1, y1, x1 + tw, y1 + th))

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
