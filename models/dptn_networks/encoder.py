import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPAINResnetBlock
from models.spade_networks.normalization import get_nonspade_norm_layer
from models.dptn_networks.PTM import TPM
import math
from models.dptn_networks import encoder
import numpy as np

class NoiseEncoder(BaseNetwork):
    def __init__(self, opt):
        super(NoiseEncoder, self).__init__()
        self.opt = opt

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        # Texture Encoder layers
        self.Texture_encoder = GetEncoder(opt, 'texture')
        self.Pose_encoder = GetEncoder(opt, 'pose')
        self.noise_mapping = MappingNetwork(z_dim=opt.z_dim, w_dim=1024, hidden_dim=opt.z_dim)

        self.head = norm_layer(nn.Conv2d(1, ndf, kw, stride=1, padding=pw))

        self.spain_layer1 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer2 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer3 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer4 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer5 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer6 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer7 = SPAINResnetBlock(ndf, ndf, opt, ndf)
        self.spain_layer8 = SPAINResnetBlock(ndf, ndf, opt, ndf)

    def forward(self, texture, pose):
        if texture.size(2) != 256 or texture.size(3) != 256:
            texture = F.interpolate(texture, size=(256, 256), mode='bilinear')

        #  texture/pose encoding
        texture = self.Texture_encoder(texture) # (b, ndf, 32, 32)
        pose = self.Pose_encoder(pose) # (b, ndf, 32, 32)


        # [Noise, Pose] encoding
        b, c, h_, w_ = pose.size()
        z = torch.randn((b, self.opt.z_dim), device=pose.device)
        w = self.noise_mapping(z)

        w = w.view(b, 1, h_, w_) # (b, 1, 32, 32)
        w = self.head(w) # (b, ndf, 32, 32)

        w = self.spain_layer1(w, pose, texture)
        w = self.spain_layer2(w, pose, texture)
        w = self.spain_layer3(w, pose, texture)
        w = self.spain_layer4(w, pose, texture)
        w = self.spain_layer5(w, pose, texture)
        w = self.spain_layer6(w, pose, texture)
        w = self.spain_layer7(w, pose, texture)
        w = self.spain_layer8(w, pose, texture)

        return w

class GetEncoder(BaseNetwork) :
    def __init__(self, opt, type):
        super(GetEncoder, self).__init__()
        self.opt = opt

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        if type == 'texture' :
            input_nc = opt.image_nc
        elif type == 'pose' :
            input_nc = opt.pose_nc
        else :
            raise Exception('Invalid encoder type')

        self.layer1 = norm_layer(nn.Conv2d(input_nc, ndf, kw, stride=2, padding=pw)) # 256x256 -> 128x128
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw)) # 128x128 -> 64x64
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw)) # 64x64 -> 32x32
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=pw)) # 32x32 -> 32x32
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=pw))  # 32x32 -> 32x32
        self.layer6 = norm_layer(nn.Conv2d(ndf * 4, ndf * 2, kw, stride=1, padding=pw))  # 32x32 -> 32x32
        self.layer7 = norm_layer(nn.Conv2d(ndf * 2, ndf * 1, kw, stride=1, padding=pw))  # 32x32 -> 32x32

        self.actvn = nn.LeakyReLU(0.2, False)
    def forward(self, x) :
        # TODO: Market1501 고려해서 나중에 수정해야함.
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        x = self.layer7(self.actvn(x))
        x = self.actvn(x)

        return x


class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=256, hidden_dim=2048):
        super(MappingNetwork, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.hidden_dim = hidden_dim

        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, z):
        w = self.mapping(z)
        return w