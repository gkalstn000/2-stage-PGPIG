"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn

from models.dptn_networks import define_En_c, define_De
from models.dptn_networks.base_network import BaseNetwork
from models.dptn_networks import encoder
from models.dptn_networks import decoder
from models.dptn_networks import PTM
import math

class DPTNGenerator(BaseNetwork):
    """
    Dual-task Pose Transformer Network (DPTN)
    :param image_nc: number of channels in input image
    :param pose_nc: number of channels in input pose
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param layers: down and up sample layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    :param output_nc: number of channels in output image
    :param num_blocks: number of ResBlocks
    :param affine: affine in Pose Transformer Module
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--activation', type=str, default='LeakyReLU', help='type of activation function')
        parser.add_argument('--type_En_c', type=str, default='default', help='selects En_c type to use for generator (default | spade | spadeattn)')
        parser.add_argument('--type_Dc', type=str, default='default', help='selects Dc type to use for generator (default | spade | spadeattn)')

        parser.set_defaults(use_spect_g=True)
        parser.set_defaults(use_coord=False)
        parser.set_defaults(norm='instance')
        parser.set_defaults(img_f=512)
        return parser
    def __init__(self, opt):
        super(DPTNGenerator, self).__init__()
        self.opt = opt
        # Encoder En_c
        # self.En_c = encoder.DefaultEncoder(opt)
        self.En_c = define_En_c(opt)
        # Pose Transformer Module (PTM)
        self.PTM = PTM.PoseTransformerModule(opt=opt)
        # SourceEncoder En_s
        self.En_s = encoder.SourceEncoder(opt)
        # OutputDecoder De
        self.De = define_De(opt)

        # self.time_embed = nn.Sequential(
        #     nn.Linear(self.time_emb_channels, conf.embed_channels),
        #     nn.SiLU(),
        #     nn.Linear(conf.embed_channels, conf.embed_channels),
        # )

    def get_pose_encoder(self, step):
        return self.positional_encoding(step)

    def forward(self,
                ref_image, ref_map,
                input_image, input_map, input_timestep):
        b, c, h, w = ref_image.size()
        time_emb = timestep_embedding(input_timestep).to(ref_image.device).view(b, 1, h, w)
        # Encode source-to-source
        F_s_s = self.En_c(ref_image+time_emb, ref_map)
        # Encode source-to-target
        F_s_t = self.En_c(input_image+time_emb, input_map)

        # Source Image Encoding
        F_s = self.En_s(ref_image+time_emb)
        # Pose Transformer Module for Dual-task Correlation
        F_s_t, _, _ = self.PTM(F_s_s, F_s_t, F_s)
        # Source-to-source Decoder (only for training)
        out_image_s = self.De(F_s_s)
        # Source-to-target Decoder
        out_image_t = self.De(F_s_t)

        return out_image_t, out_image_s


def timestep_embedding(timesteps, dim=256*176, max_period=1000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                   torch.arange(start=0, end=half, dtype=torch.float32) /
                   half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding