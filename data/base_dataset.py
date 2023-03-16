"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
import random
import json
import time

import util.util as util
import os

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.phase = opt.phase


        transform_list=[]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

        if isinstance(opt.load_size, int):
            self.load_size = (opt.load_size, opt.load_size)
        else:
            self.load_size = opt.load_size

        self.cum_time = 0
        self.count = 0


    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        for i in trange(size, desc = 'Loading data pairs ...'):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def __getitem__(self, index):
        start = time.time()
        P1_name, P2_name = self.name_pairs[index]
        PC_name = f'{P1_name.replace(".jpg", "")}_2_{P2_name.replace(".jpg", "")}_vis.jpg'

        P1_path = os.path.join(self.image_dir, P1_name) # person 1
        P2_path = os.path.join(self.image_dir, P2_name) # person 2
        # Canonical_path = os.path.join(self.canonical_dir, PC_name)

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')
        # Canonical_img = Image.open(Canonical_path).convert('RGB')

        P1_img = F.resize(P1_img, self.load_size)
        P2_img = F.resize(P2_img, self.load_size)
        # Canonical_img = F.resize(Canonical_img, self.load_size)

        # T_ST, T_ST_inv = self.calculate_transformation_matrix(P1_name, P2_name)
        # P1 preprocessing
        P1 = self.trans(P1_img)
        BP1 = self.obtain_bone(P1_name)
        # BP1 = torch.load(os.path.join(self.opt.dataroot, f'{self.phase}_map', P1_name.replace('jpg', 'pt')))[:self.opt.pose_nc]
        # P2 preprocessing
        P2 = self.trans(P2_img)
        BP2 = self.obtain_bone(P2_name)
        # BP2 = torch.load(os.path.join(self.opt.dataroot, f'{self.phase}_map', P2_name.replace('jpg', 'pt')))[:self.opt.pose_nc]

        if self.opt.pos_encoding :
            BP1_pos = self.obtain_bone_pos(P1_name)
            BP1 = BP1 + BP1_pos
            # BP1_max, BP1_min = BP1.view(self.opt.pose_nc, -1).max(-1, keepdims=True)[0], BP1.view(self.opt.pose_nc, -1).min(-1, keepdims=True)[0]
            # BP1 = (BP1 + 1) / 3

            BP2_pos = self.obtain_bone_pos(P2_name)
            BP2 = BP2 + BP2_pos
            # BP2_max, BP2_min = BP2.view(self.opt.pose_nc, -1).max(-1, keepdims=True)[0], BP2.view(self.opt.pose_nc, -1).min(-1, keepdims=True)[0]
            # BP2 = (BP2 + 1) / 3
        # Canonical_img
        # PC = self.trans(Canonical_img)
        # BPC = self.obtain_bone(PC_name)
        # BPC = torch.load(os.path.join(self.opt.dataroot, 'canonical_map.pt'))[:self.opt.pose_nc]
        # BPC = torch.load(os.path.join(self.opt.dataroot, f'{self.phase}_map', self.annotation_file_canonical.loc[PC_name].item()))[:self.opt.pose_nc]


        input_dict = {'src_image' : P1,
                      'src_map': BP1,
                      'tgt_image' : P2,
                      'tgt_map' : BP2,
                      'canonical_image' : P2,
                      'canonical_map' : BP2,
                      'path' : PC_name}

        # num_worker optimizing
        # end = time.time()
        # self.cum_time += end - start
        # self.count += 1
        # if self.count == self.opt.batchSize :
        #     print(f'{self.cum_time}')
        #     self.cum_time = 0
        #     self.count = 0
        # ====================
        return input_dict


    def obtain_bone_pos(self, name):
        y, x = self.annotation_file.loc[name]
        coord = util.make_coord_array(y, x)

        relative_pos_matrix = []
        for h, w in coord :
            if (h <= -1 or w <= -1) or (not (0<= h < self.opt.old_size[0] and 0<=w<self.opt.old_size[1])) :
                relative_pos_matrix.append(torch.zeros(1, self.opt.load_size, self.opt.load_size))
                continue
            h = int(h / self.opt.old_size[0] * self.opt.load_size)
            w = int(w / self.opt.old_size[1] * self.opt.load_size)

            h_index = self.opt.load_size - h
            w_index = self.opt.load_size - w
            matrix = self.positional_matrix[:, h_index: h_index + self.opt.load_size, w_index: w_index + self.opt.load_size]
            assert matrix.shape == (1, self.opt.load_size, self.opt.load_size), print(f'({h_index}, {w_index}) / ({h, w})')
            relative_pos_matrix.append(matrix)
        relative_pos_matrix = torch.concatenate(relative_pos_matrix)
        return relative_pos_matrix
    def obtain_bone(self, name):
        if '_2_' in name :
            y, x = self.annotation_file_canonical.loc[name]
        else :
            y, x = self.annotation_file.loc[name]
        coord = util.make_coord_array(y, x)
        return self.obtain_bone_with_coord(coord)
    def obtain_bone_with_coord(self, coord):
        # Keypoint map
        keypoint = util.cords_to_map(coord, self.opt)
        keypoint = np.transpose(keypoint, (2, 0, 1))
        keypoint = torch.Tensor(keypoint)
        if self.opt.pose_nc == 18 :
            return keypoint
        # Limb map
        limb = util.limbs_to_map(coord, self.opt)
        limb = np.transpose(limb, (2, 0, 1))
        limb = torch.Tensor(limb)

        return torch.cat([keypoint, limb])
    def get_canonical_pose(self):
        return ['[28, 54, 54, 93, 130, 55, 95, 131, 117, 180, 233, 117, 178, 230, 24, 23, 27, 26]',
                '[88, 88, 67, 66, 63, 108, 111, 119, 78, 82, 81, 103, 100, 91, 84, 93, 77, 100]']

    def coord_to_PIL(self, coord) :
        heatmap = self.obtain_bone_with_coord(coord)
        heatmap = util.map_to_img(heatmap)
        heatmap_array = (heatmap.numpy() * 255).astype(np.uint8)
        return Image.fromarray(heatmap_array)
def get_transform(B, T):
    B_S_transformed = np.matmul(np.hstack([B, np.ones((B.shape[0], 1))]), T.T)
    B_S_transformed = B_S_transformed[:, :2] / B_S_transformed[:, 2:]
    return B_S_transformed
def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
