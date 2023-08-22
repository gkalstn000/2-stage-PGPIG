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
        mult = opt.mult

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = 2 * opt.pose_nc + opt.image_nc

        for i in range(self.layers):
            mult_prev = mult
            mult = min(2 ** (self.layers - i - 2), opt.img_f // opt.ngf) if i != self.layers - 1 else 1
            down = SPADEResnetBlock(nf * mult_prev, nf * mult, opt, 'decoder')
            setattr(self, 'decoder' + str(i), down)

        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, texture_information):
        texture_information = torch.cat(texture_information, 1)

        for i in range(self.layers):
            model = getattr(self, 'decoder' + str(i))
            x = model(x, texture_information)
            x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x
class DefaultDecoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultDecoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        mult = 4
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)

        self.res_block1 = modules.ResBlockDecoder(opt.ngf * 4, opt.ngf * 2, opt.ngf * mult, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.res_block2 = modules.ResBlockDecoder(opt.ngf * 2, opt.ngf * 1, opt.ngf * mult, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.res_block3 = modules.ResBlockDecoder(opt.ngf * 1, opt.ngf * 1, opt.ngf * mult, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.t_block0  = nn.Sequential(nn.Linear(256, opt.ngf * 4),
                                       nn.SiLU(),
                                       nn.Linear(opt.ngf * 4, opt.ngf * 4),
                                       )

        self.t_block1 = nn.Sequential(nn.Linear(256, opt.ngf * 2),
                                      nn.SiLU(),
                                      nn.Linear(opt.ngf * 2, opt.ngf * 2),
                                      )
        self.t_block2 = nn.Sequential(nn.Linear(256, opt.ngf * 2),
                                      nn.SiLU(),
                                      nn.Linear(opt.ngf * 2, opt.ngf * 2),
                                      )


        self.outconv = modules.Output(opt.ngf, opt.output_nc, 3, None, nonlinearity, opt.use_spect_g, opt.use_coord)
    def forward(self, x, time_emb):
        out = self.res_block1(x)
        out = self.apply_conditions(out, self.t_block0(time_emb))
        out = self.res_block2(out)
        out = self.apply_conditions(out, self.t_block1(time_emb))
        out = self.res_block3(out)
        out = self.apply_conditions(out, self.t_block2(time_emb))
        out = self.outconv(out)
        return out