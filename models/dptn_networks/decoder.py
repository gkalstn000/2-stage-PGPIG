import torch
import torch.nn as nn
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock
from models.spade_networks.normalization import get_nonspade_norm_layer
from models.dptn_networks.modules import ResBlockDecoder
import torch.nn.functional as F

class SpadeDecoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        norm_nc = opt.pose_nc

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt, norm_nc)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt, norm_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, norm_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt, norm_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt, norm_nc)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt, norm_nc)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, z, texture_information):
        texture_information = torch.cat(texture_information, 1)

        x = self.fc(z)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)

        x = self.head_0(x, texture_information)

        x = self.up(x)
        x = self.G_middle_0(x, texture_information)
        x = self.G_middle_1(x, texture_information)

        x = self.up(x)
        x = self.up_0(x, texture_information)
        x = self.up(x)
        x = self.up_1(x, texture_information)
        x = self.up(x)
        x = self.up_2(x, texture_information)
        x = self.up(x)
        x = self.up_3(x, texture_information)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / (opt.load_size[1] / opt.load_size[0]))

        return sw, sh



class DefaultDecoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultDecoder, self).__init__()
        self.opt = opt
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.fc = nn.Sequential(nn.Linear(opt.z_dim, 16 * ndf * self.sw * self.sh),
                                nn.InstanceNorm1d(num_features=self.opt.z_dim),
                                nn.Dropout(p=0.2))

        self.head_0 = norm_layer(nn.Conv2d(16 * ndf, 16 * ndf, 3, stride=1, padding=1))
        self.G_middle_0 = norm_layer(nn.Conv2d(16 * ndf, 16 * ndf, 3, stride=1, padding=1))
        self.G_middle_1 = norm_layer(nn.Conv2d(16 * ndf, 16 * ndf, 3, stride=1, padding=1))

        self.up_0 = ResBlockDecoder(16 * ndf, 8 * ndf)
        self.up_1 = ResBlockDecoder(8 * ndf, 4 * ndf)
        self.up_2 = ResBlockDecoder(4 * ndf, 2 * ndf)
        self.up_3 = ResBlockDecoder(2 * ndf, 1 * ndf)

        final_nc = ndf


        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)

        x = F.leaky_relu(self.head_0(x), 2e-1)
        x = self.up(x)
        x = F.leaky_relu(self.G_middle_0(x), 2e-1)
        x = F.leaky_relu(self.G_middle_1(x), 2e-1)

        x = self.up_0(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.up_3(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / (opt.load_size[1] / opt.load_size[0]))

        return sw, sh