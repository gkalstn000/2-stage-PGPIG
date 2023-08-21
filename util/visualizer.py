"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import ntpath
import time
import numpy as np

from . import util
import numpy as np
from PIL import Image
import torch
import torchvision

import wandb

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

class Visualizer():
    def __init__(self, opt):
        wandb.login()
        wandb.init(project="DPTN", name=opt.id, settings=wandb.Settings(code_dir="."), resume=opt.continue_train)
        self.wandb = wandb
        self.opt = opt
        self.tf_log = not opt.no_log
        self.win_size = opt.display_winsize
        self.display_num = 8
        self.id = opt.id
        if self.tf_log:
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.id, 'logs')

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.id, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):
        if self.tf_log:  # show images in tensorboard output
            for label, image_tensor in visuals.items():
                image_tensor = (image_tensor[:self.display_num] + 1) / 2
                image_tensor.clamp_(0, 1)
                img_grid = torchvision.utils.make_grid(image_tensor, nrow=4, padding=0, normalize=False).permute(1, 2, 0)
                image = Image.fromarray((img_grid.numpy() * 255).astype(np.uint8))
                self.wandb.log({label: self.wandb.Image(image)})

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            self.wandb.log(errors)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals, display_num = 3):
        # image shape : (B, C, H, W)
        for key, t in visuals.items():
            t = t[:display_num]
            tile = self.opt.batchSize > 1

            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, visuals, epoch, step):
        visuals = self.convert_visuals_to_numpy(visuals)

        for label, image_numpy in visuals.items():
            save_path = self._get_save_path(label, epoch, step)
            util.save_image(image_numpy, save_path, create_dir=True)

    def _get_save_path(self, image_name, epoch, iteration, ext='png'):
        subdir_path = os.path.join(self.opt.checkpoints_dir, self.opt.id)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(
            subdir_path, '%s_epoch_{:05}_iteration_{:09}.{}'.format(
                image_name, epoch, iteration, ext))