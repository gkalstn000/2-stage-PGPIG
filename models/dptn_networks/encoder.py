import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock
from models.spade_networks.normalization import get_nonspade_norm_layer
from models.dptn_networks.PTM import TPM
import math
from models.dptn_networks import encoder
import numpy as np

class SpadeEncoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = opt.image_nc
        norm_nc = opt.pose_nc
        self.head_0 = SPADEResnetBlock(input_nc, nf, opt, norm_nc)

        self.mult = 1
        for i in range(self.layers - 1) :
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = SPADEResnetBlock(opt.ngf * mult_prev, opt.ngf * self.mult, opt, norm_nc)
            setattr(self, 'down' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

        self.down = nn.MaxPool2d(2, stride=2)

        self.mu = nn.Conv2d(opt.ngf * self.mult, opt.ngf * self.mult, 3, stride=1, padding=1)
        self.var = nn.Conv2d(opt.ngf * self.mult, opt.ngf * self.mult, 3, stride=1, padding=1)
    def forward(self, texture, bone):

        x = self.head_0(texture, bone)
        x = self.down(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'down' + str(i))
            x = model(x, bone)
            x = self.down(x)

        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return self.mu(x), self.var(x)

class NoiseEncoder(BaseNetwork):
    def __init__(self, opt):
        super(NoiseEncoder, self).__init__()
        self.opt = opt

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        # Texture Encoder layers
        input_nc = opt.image_nc
        self.TClayer1 = norm_layer(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pw))
        self.TClayer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.TClayer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.TClayer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=2, padding=pw))

        self.TFlayer = nn.Sequential(nn.Linear(256, self.opt.z_dim),
                                     nn.InstanceNorm1d(num_features=self.opt.z_dim),
                                     nn.Dropout(p=0.2))
        # Pose Encoder layers
        input_nc = opt.pose_nc
        self.PClayer1 = norm_layer(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pw))
        self.PClayer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.PClayer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.PClayer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=2, padding=pw))
        self.PFlayer = nn.Sequential(nn.Linear(256, self.opt.z_dim),
                                     nn.InstanceNorm1d(num_features=self.opt.z_dim),
                                     nn.Dropout(p=0.2))


        # [Noise, Pose] encoder
        self.NPlayer = nn.Sequential(nn.Linear(512, self.opt.z_dim),
                                     nn.InstanceNorm1d(num_features=self.opt.z_dim),
                                     nn.Dropout(p=0.2))

        # Texture-Pose attn module
        self.TPM = TPM(ndf * 4, 2)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.so = s0 = 4

        self.final_nc = nn.Sequential(nn.Linear(self.opt.z_dim * self.opt.z_dim, ndf * 8 * s0 * s0),
                                      nn.InstanceNorm1d(num_features=ndf * 8 * s0 * s0))

    def forward(self, texture, pose):
        if texture.size(2) != 256 or texture.size(3) != 256:
            texture = F.interpolate(texture, size=(256, 256), mode='bilinear')

        # Texture 'x' encoding
        texture = self.TClayer1(texture)              # 256x256 -> 128x128
        texture = self.TClayer2(self.actvn(texture))  # 128x128 -> 64x64
        texture = self.TClayer3(self.actvn(texture))  # 64x64 -> 32x32
        texture = self.TClayer4(self.actvn(texture))  # 32x32 -> 16x16
        texture = self.actvn(texture)
        b, c, h, w = texture.size()
        texture = texture.view(b, c, h*w)  # 16x16 -> 256
        texture = self.TFlayer(texture) # b, c, 256

        # Pose 'c' encoding
        pose = self.PClayer1(pose)              # 256x256 -> 128x128
        pose = self.PClayer2(self.actvn(pose))  # 128x128 -> 64x64
        pose = self.PClayer3(self.actvn(pose))  # 64x64 -> 32x32
        pose = self.PClayer4(self.actvn(pose))  # 32x32 -> 16x16
        pose = self.actvn(pose)
        pose = pose.view(b, c, h*w)  # 16x16 -> 256
        pose = self.PFlayer(pose) # b, c, 256

        # [Noise, Pose] encoding
        noise = torch.randn((b, c, self.opt.z_dim), device=pose.device)
        pose = torch.cat([noise, pose], -1)
        pose = self.NPlayer(pose)

        x = self.TPM(pose, texture)

        x = x.view(x.size(0), -1)
        x = self.actvn(self.final_nc(x))

        return x



class SourceEncoder(nn.Module):
    """
    Source Image Encoder (En_s)
    :param image_nc: number of channels in input image
    :param ngf: base filter channel
    :param img_f: the largest feature channels
    :param encoder_layer: encoder layers
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """

    def __init__(self, opt):
        super(SourceEncoder, self).__init__()
        self.opt = opt
        self.encoder_layer = opt.layers_g

        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        input_nc = opt.image_nc

        self.block0 = modules.EncoderBlockOptimized(input_nc, opt.ngf, norm_layer,
                                                    nonlinearity, opt.use_spect_g, opt.use_coord)
        mult = 1
        for i in range(opt.layers_g - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = modules.EncoderBlock(opt.ngf * mult_prev, opt.ngf * mult, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'encoder' + str(i), block)

        if isinstance(opt.load_size, int) :
            h = w = opt.load_size
        else :
            h, w = opt.load_size
        self.ch = h // 2**(mult-1)
        self.cw = w // 2**(mult-1)
        self.mu = nn.Linear(opt.ngf * mult * self.ch * self.cw, self.ch*self.cw)
        self.var = nn.Linear(opt.ngf * mult * self.ch * self.cw, self.ch*self.cw)

        self.mult = mult
# (ndf * 8 * s0 * s0, 256)
    def forward(self, x):
        out = self.block0(x)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = out.view(x.size(0), -1)
        return self.mu(out), self.var(out)



class DefaultEncoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultEncoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        input_nc = 2 * opt.pose_nc + opt.image_nc

        self.block0 = modules.EncoderBlockOptimized(input_nc, opt.ngf, norm_layer,
                                                    nonlinearity, opt.use_spect_g, opt.use_coord)
        self.mult = 1
        for i in range(self.layers - 1):
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = modules.EncoderBlock(opt.ngf * mult_prev, opt.ngf * self.mult, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'encoder' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

    def forward(self, x, texture_information):
        # Source-to-source Encoder
        x = self.block0(x) # (B, C, H, W) -> (B, ngf, H/2, W/2)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            x = model(x)
        # input_ size : (B, ngf * 2^2, H/2^layers, C/2^layers)
        # Source-to-source Resblocks
        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x