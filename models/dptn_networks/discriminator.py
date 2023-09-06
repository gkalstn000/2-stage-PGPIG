import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import numpy as np
from models.dptn_networks.base_network import BaseNetwork
from models.dptn_networks import modules
import torch
class ResDiscriminator(BaseNetwork):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param use_spect: use spectual normalization
    :param use_coord: use coordConv operation
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(use_spect_d=True)
        return parser

    def __init__(self, opt, input_nc=3, ndf=32, img_f=128, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()
        self.opt = opt
        self.layers = opt.dis_layers
        norm_layer = modules.get_norm_layer(norm_type=norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # encoder part
        self.block0 = modules.ResBlockEncoderOptimized(input_nc, ndf, ndf, norm_layer, nonlinearity, opt.use_spect_d, use_coord)

        mult = 1
        for i in range(opt.dis_layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf)
            block = modules.ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, opt.use_spect_d, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1))
        self.fc_step = nn.Sequential(nn.Linear(ndf*mult * 16 * 11, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, opt.step_size),
                                     )
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) # (B, 128, 16, 11)
        b, c, h, w = out.size()
        step = self.fc_step(out.view(b, -1))
        pred = self.conv(self.nonlinearity(out))
        return pred, self.log_softmax(step)






class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = NLayerDiscriminator
        # subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
        #                                     'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

        self.fc_step = nn.Sequential(nn.Linear(35 * 25 + 19 * 14, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, opt.step_size),
                                     )
        self.log_softmax = nn.LogSoftmax(dim=1)
    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            if 'fc' in name : break
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        b, c, h, w = input.size()
        step_input = torch.cat([result[0][-1].view(b, -1), result[1][-1].view(b, -1)], -1)
        step = self.fc_step(step_input)

        return result, self.log_softmax(step)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # parser.add_argument('--dis_layers', type=int, default=4,
        #                     help='# layers in each discriminator')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')

        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, norm_type=opt.norm)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.dis_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.dis_layers - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                    stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self, opt):
        input_nc = opt.pose_nc + opt.image_nc

        return input_nc

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.opt.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]

def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        layer = SpectralNorm(layer)
        subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)

        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer