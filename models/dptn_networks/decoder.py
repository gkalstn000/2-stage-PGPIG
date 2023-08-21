import torch
import torch.nn as nn
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock

import torch.nn.functional as F

class SpadeDecoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        mult = 4

        self.decoder1 = SPADEResnetBlock(nf * 4, nf * 2, nf * 4)
        self.up1 = nn.ConvTranspose2d(nf * 2, nf * 2, 2, stride=2)
        self.decoder2 = SPADEResnetBlock(nf * 2, nf * 1, nf * 2)
        self.up2 = nn.ConvTranspose2d(nf * 1, nf * 1, 2, stride=2)
        self.decoder3 = SPADEResnetBlock(nf * 1, nf * 1, nf * 1)
        self.up3 = nn.ConvTranspose2d(nf * 1, nf * 1, 2, stride=2)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, x, texture_information):
        e1, e2, e3 = texture_information

        x = self.decoder1(x, e3)
        x = self.up1(x)
        x = self.decoder2(x, e2)
        x = self.up2(x)
        x = self.decoder3(x, e1)
        x = self.up3(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
class DefaultDecoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultDecoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        mult = opt.mult
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)

        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), opt.img_f // opt.ngf) if i != self.layers - 1 else 1
            up = modules.ResBlockDecoder(opt.ngf * mult_prev, opt.ngf * mult, opt.ngf * mult, norm_layer,
                                 nonlinearity, opt.use_spect_g, opt.use_coord)
            setattr(self, 'decoder' + str(i), up)
        self.outconv = modules.Output(opt.ngf, opt.output_nc, 3, None, nonlinearity, opt.use_spect_g, opt.use_coord)
    def forward(self, x, texture_information):
        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            x = model(x)
        out = self.outconv(x)
        return out