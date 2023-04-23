"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.spade_networks.normalization import SPADE, AdaIN


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPAINResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, norm_nc):
        '''
        :param fin: input dim of main feature map
        :param fout: output dim of main feature map
        :param opt: options
        :param norm_nc: norm input dim
        '''
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # SPADE
        self.spade_conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.spade_conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.spade_conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_config_str = opt.norm_G.replace('spectral', '')
        self.spade_norm_0 = SPADE(spade_config_str, fin, norm_nc)
        self.spade_norm_1 = SPADE(spade_config_str, fmiddle, norm_nc)
        if self.learned_shortcut:
            self.spade_norm_s = SPADE(spade_config_str, fin, norm_nc)

        # ADAIN
        self.adain_conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.adain_conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut :
            self.adain_conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.adain_norm = AdaIN()



    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, pose_information, texture_information):
        # SPADE Residual part
        x_spade = self.spade_conv_0(self.actvn(self.spade_norm_0(x, pose_information)))
        x_spade = self.spade_conv_1(self.actvn(self.spade_norm_1(x_spade, pose_information)))

        # ADAIN Residual part
        x_adain = self.adain_conv_0(self.actvn(self.adain_norm(x, texture_information)))
        x_adain = self.adain_conv_1(self.actvn(self.adain_norm(x_adain, texture_information)))
        out = x_spade + x_adain
        return x + out

    def spade_shortcut(self, x, pose_information):
        if self.learned_shortcut:
            x_s = self.spade_conv_s(self.spade_norm_s(x, pose_information))
        else:
            x_s = x
        return x_s
    def adain_shortcut(self, x, texture_information):
        if self.learned_shortcut:
            x_s = self.adain_conv_s(self.adain_norm(x, texture_information))
        else:
            x_s = x
        return x_s
    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


if __name__ == "__main__" :
    conv0 = nn.Conv2d(10, 10, kernel_size=3, padding=1)