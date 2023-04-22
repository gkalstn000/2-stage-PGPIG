import copy
import torch
from torch import nn
from models.dptn_networks import modules
import math

class PoseTransformerModule(nn.Module):
    """
    Pose Transformer Module (PTM)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param num_CABs: number of CABs
    :param num_TTBs: number of TTBs
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, opt):
        super(PoseTransformerModule, self).__init__()
        self.opt = opt
        d_model = opt.ngf * opt.mult
        encoder_layer = CAB(d_model, opt.nhead, d_model,
                                                opt.activation, opt.affine, opt.norm)
        if opt.norm == 'batch':
            encoder_norm = None
            decoder_norm = nn.BatchNorm1d(d_model, affine=opt.affine)
        elif opt.norm == 'instance':
            encoder_norm = None
            decoder_norm = nn.InstanceNorm1d(d_model, affine=opt.affine)

        self.encoder = CABs(encoder_layer, opt.num_CABs, encoder_norm)

        decoder_layer = TTB(d_model, opt.nhead, d_model,
                                                opt.activation, opt.affine, opt.norm)

        self.decoder = TTBs(decoder_layer, opt.num_TTBs, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = opt.nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, val, pos_embed=None):
        # src: key
        # tgt: query
        # val: value
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        tgt = tgt.flatten(2).permute(2, 0, 1)
        val = val.flatten(2).permute(2, 0, 1)
        if pos_embed != None:
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, pos=pos_embed)
        hs, first_attn_weights, last_attn_weights = self.decoder(tgt, memory, val, pos=pos_embed) # query, key, value 순서
        return hs.view(bs, c, h, w), first_attn_weights, last_attn_weights


class CABs(nn.Module):
    """
    Context Augment Blocks (CABs)
    :param encoder_layer: CAB
    :param num_CABS: number of CABs
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, encoder_layer, num_CABs, norm=None):
        super(CABs, self).__init__()
        self.layers = _get_clones(encoder_layer, num_CABs)
        self.norm = norm

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        if self.norm is not None:
            output = self.norm(output.permute(1, 2, 0)).permute(2, 0, 1)

        return output


class TTBs(nn.Module):
    """
    Texture Transfer Blocks (TTBs)
    :param decoder_layer: TTB
    :param num_layers: number of TTBs
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, decoder_layer, num_TTBs, norm=None):
        super(TTBs, self).__init__()
        self.layers = _get_clones(decoder_layer, num_TTBs)
        self.norm = norm

    def forward(self, tgt, memory, val, pos = None):
        output = tgt
        weight_list = []
        for layer in self.layers:
            output, attn_output_weights = layer(output, memory, val, pos=pos)
            weight_list.append(attn_output_weights.cpu())

        if self.norm is not None:
            output = self.norm(output.permute(1, 2, 0))
        return output, weight_list[0], weight_list[-1]


class CAB(nn.Module):
    """
    Context Augment Block (CAB)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 activation="LeakyReLU", affine=True, norm='instance'):
        super(CAB, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm2 = nn.BatchNorm1d(d_model, affine=affine)
        else:
            self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)

        self.activation = modules.get_nonlinearity_layer(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + src2
        src = self.norm1(src.permute(1, 2, 0)).permute(2, 0, 1)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src.permute(1, 2, 0)).permute(2, 0, 1)
        return src


class TTB(nn.Module):
    """
    Texture Transfer Block (TTB)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 activation="LeakyReLU", affine=True, norm='instance'):
        super(TTB, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm2 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm3 = nn.BatchNorm1d(d_model, affine=affine)
        else:
            self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm3 = nn.InstanceNorm1d(d_model, affine=affine)

        self.activation = modules.get_nonlinearity_layer(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, val, pos = None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2, attn_output_weights = self.multihead_attn(query=self.with_pos_embed(tgt, pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=val)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        return tgt, attn_output_weights


class CrossAttnModule(nn.Module) :
    def __init__(self, d_model, nhead, dim_feedforward=2048, affine=True):
        super(CrossAttnModule, self).__init__()

        self.attn_layer = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.fc_layer = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                      nn.LeakyReLU(0.2, False),
                                      nn.Linear(dim_feedforward, d_model),
                                      nn.Dropout(p=0.2))

        self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
        self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)
        self.actvn = nn.LeakyReLU(0.2, False)

    def forward(self, query, key, value):
        '''
        :param query: (b, n, c)
        :param key:  (b, n, c)
        :param value:  (b, n, c)
        :return:
        '''
        attn_output, _ = self.attn_layer(query, key, value)
        x = value + attn_output
        x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.actvn(x)

        ff_output = self.actvn(self.fc_layer(x))
        x = x + ff_output
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.actvn(x)

        return x
class TPM(nn.Module):
    """
    Texture Transfer Block (TTB)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, affine=True):
        super(TPM, self).__init__()
        self.layer1 = CrossAttnModule(d_model, nhead, dim_feedforward, affine)
        self.layer2 = CrossAttnModule(d_model, nhead, dim_feedforward, affine)
        self.layer3 = CrossAttnModule(d_model, nhead, dim_feedforward, affine)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pos_encoder = PositionalEncoding(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else self.pos_encoder(tensor)

    def forward(self, query, KV, pos = True):
        '''
        :param query: (b, c, n)
        :param key:  (b, c, n)
        :param value:  (b, c, n)
        :param pos: (1, c, n)
        :return:
        '''
        query = query.permute(0, 2, 1)
        KV = KV.permute(0, 2, 1)

        query = self.with_pos_embed(query, pos)
        key = value = self.with_pos_embed(KV, pos)

        x = self.layer1(query, key, value)
        x = self.layer2(query, key, x)
        x = self.layer3(query, key, x)

        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


