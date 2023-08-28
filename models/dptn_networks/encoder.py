import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dptn_networks import modules
from models.dptn_networks.base_network import BaseNetwork
from models.spade_networks.architecture import SPADEResnetBlock

class SpadeAttnEncoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = 2 * opt.pose_nc + opt.image_nc

        self.head_0 = SPADEResnetBlock(input_nc, nf, opt, 'encoder')

        self.mult = 1
        for i in range(self.layers - 1) :
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = SPADEResnetBlock(opt.ngf * mult_prev, opt.ngf * self.mult, opt, 'encoder')
            setattr(self, 'down' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

        self.down = nn.MaxPool2d(2, stride=2)

    def forward(self, x, texture_information):
        x = self.head_0(x, texture_information)
        x = self.down(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'down' + str(i))
            x = model(x, texture_information)
            x = self.down(x)

        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x

class SpadeEncoder(BaseNetwork) :
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.layers = opt.layers_g
        nf = opt.ngf
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        input_nc = 2 * opt.pose_nc + opt.image_nc

        self.head_0 = SPADEResnetBlock(input_nc, nf, opt, 'encoder')

        self.mult = 1
        for i in range(self.layers - 1) :
            mult_prev = self.mult
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            block = SPADEResnetBlock(opt.ngf * mult_prev, opt.ngf * self.mult, opt, 'encoder')
            setattr(self, 'down' + str(i), block)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)

        self.down = nn.MaxPool2d(2, stride=2)
    def forward(self, bone1, bone2, img2, texture_information):
        texture_information = torch.cat(texture_information, 1)
        x = torch.cat([img2, bone2, bone1], 1)

        x = self.head_0(x, texture_information)
        x = self.down(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'down' + str(i))
            x = model(x, texture_information)
            x = self.down(x)

        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x

class AttnEncoder(BaseNetwork):
    def __init__(self, opt):
        super(AttnEncoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)

        self.image_encoder = nn.ModuleList()
        self.bone_encoder = nn.ModuleList()

        in_channel_image = opt.image_nc
        in_channel_bone = opt.pose_nc
        out_channel = opt.ngf
        for i in range(self.layers):
            self.image_encoder.append(modules.EncoderBlock(in_channel_image, out_channel, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord))
            self.bone_encoder.append(modules.EncoderBlock(in_channel_bone, out_channel, norm_layer,
                                         nonlinearity, opt.use_spect_g, opt.use_coord))
            self.mult = min(2 ** (i + 1), opt.img_f // opt.ngf)
            in_channel_image = out_channel
            in_channel_bone = out_channel
            out_channel = opt.ngf * self.mult
        self.mult //= 2
        self.Attn = modules.CrossAttnModule(opt.ngf * self.mult, opt.nhead, opt.ngf * self.mult)

        # ResBlocks
        for i in range(opt.num_blocks):
            block = modules.ResBlock(opt.ngf * self.mult, opt.ngf * self.mult, norm_layer=norm_layer,
                                     nonlinearity=nonlinearity, use_spect=opt.use_spect_g, use_coord=opt.use_coord)
            setattr(self, 'mblock' + str(i), block)


    def forward(self, Q, K, V):
        '''
        :param Query: Query Bone
        :param Key: Source Bone
        :param Value: Source Image
        :return:
        '''
        for layer in self.bone_encoder :
            Q = layer(Q)
            K = layer(K)
        for layer in self.image_encoder :
            V = layer(V)

        x, _ = self.Attn(Q, K, V)

        # Source-to-source Resblocks
        for i in range(self.opt.num_blocks):
            model = getattr(self, 'mblock' + str(i))
            x = model(x)
        return x

class DefaultEncoder(BaseNetwork):
    def __init__(self, opt):
        super(DefaultEncoder, self).__init__()
        self.opt = opt
        self.layers = opt.layers_g
        norm_layer = modules.get_norm_layer(norm_type=opt.norm)
        nonlinearity = modules.get_nonlinearity_layer(activation_type=opt.activation)
        input_nc = 1 * opt.pose_nc + opt.image_nc

        self.enc_block0 = modules.EncoderBlockOptimized(input_nc, opt.ngf, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.enc_block1 = modules.EncoderBlock(opt.ngf * 1, opt.ngf * 2, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.enc_block2 = modules.EncoderBlock(opt.ngf * 2, opt.ngf * 4, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.res_block1 = modules.ResBlock(opt.ngf * 4, opt.ngf * 4, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.res_block2 = modules.ResBlock(opt.ngf * 4, opt.ngf * 4, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)
        self.res_block3 = modules.ResBlock(opt.ngf * 4, opt.ngf * 4, norm_layer, nonlinearity, opt.use_spect_g, opt.use_coord)

        # self.t_block0  = nn.Sequential(nn.Linear(256, opt.ngf * 2),
        #                                nn.SiLU(),
        #                                nn.Linear(opt.ngf * 2, opt.ngf * 2),
        #                                )
        #
        # self.t_block1 = nn.Sequential(nn.Linear(256, opt.ngf * 4),
        #                               nn.SiLU(),
        #                               nn.Linear(opt.ngf * 4, opt.ngf * 4),
        #                               )
        # self.t_block2 = nn.Sequential(nn.Linear(256, opt.ngf * 8),
        #                               nn.SiLU(),
        #                               nn.Linear(opt.ngf * 8, opt.ngf * 8),
        #                               )
    def forward(self, image, bone):
        x = torch.cat([image, bone], 1)
        out = self.enc_block0(x)
        # out = self.apply_conditions(out, self.t_block0(time_emb))
        out = self.enc_block1(out)
        # out = self.apply_conditions(out, self.t_block1(time_emb))
        out = self.enc_block2(out)
        # out = self.apply_conditions(out, self.t_block2(time_emb))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)

        return out
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

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.encoder_layer - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        return out
