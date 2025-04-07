# -*- coding: utf-8 -*-
"""
@author: 
"""


import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class _ComplexLinear(nn.Module):
    """
    https://github.com/wavefrontshaping/complexPyTorch/tree/master
    """

    def __init__(self, in_features, out_features):
        super(_ComplexLinear, self).__init__()
        self.linear_real = nn.Linear(in_features, out_features)
        self.linear_imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        x_freq = torch.fft.fft(x)
        y_freq = _apply_complex(
            self.linear_real, self.linear_imag, x_freq)
        return torch.fft.ifft(y_freq).real


def _apply_complex(func_real, func_imag, x, dtype=torch.complex64):
    return ((func_real(x.real)-func_imag(x.imag)).type(dtype) +
            1j*(func_real(x.imag)+func_imag(x.real)).type(dtype))


class _BatchNorm2d(nn.Module):
    def __init__(self):
        super(_BatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, h):
        h = torch.unsqueeze(h, 1)
        h = self.bn(h)
        h = h[:, 0, :, :]
        return h


class _TransLinear(nn.Module):
    def __init__(self, in_dim, in_len, out_dim, is_preact, norm_layer):
        super(_TransLinear, self).__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_dim, out_dim)

        if norm_layer == 'BN':
            self.normalize = _BatchNorm2d()
        elif norm_layer == 'LN':
            self.normalize = nn.LayerNorm([in_dim, in_len, ])
        elif norm_layer == 'IN':
            self.normalize = nn.InstanceNorm1d(in_dim)
        else:
            self.normalize = nn.Identity()
        self.add_module('normalize', self.normalize)

        self.is_preact = is_preact

    def forward(self, x):
        is_preact = self.is_preact
        h = self.normalize(x)
        h = torch.transpose(h, 1, 2)
        if is_preact:
            h = self.relu(h)
            h = self.linear(h)
        else:
            h = self.linear(h)
        h = torch.transpose(h, 1, 2)
        return h


class _Block_1(nn.Module):
    def __init__(self, in_len, out_len, seq_dim, proj_dim, dim_factor,
                 norm_layer, is_preact, is_random, is_freq,
                 is_skip, is_first_relu):
        super(_Block_1, self).__init__()
        if is_skip:
            self.relu_0 = nn.ReLU()

        if is_freq:
            self.linear_0 = _ComplexLinear(in_len, out_len)
        else:
            self.linear_0 = nn.Linear(in_len, out_len)
        self.add_module('linear_0', self.linear_0)

        if norm_layer == 'BN':
            self.normalize = _BatchNorm2d()
        elif norm_layer == 'LN':
            self.normalize = nn.LayerNorm([seq_dim, in_len, ])
        elif norm_layer == 'IN':
            self.normalize = nn.InstanceNorm1d(seq_dim)
        else:
            self.normalize = nn.Identity()
        self.add_module('normalize', self.normalize)

        proj_dim = np.ceil(proj_dim * dim_factor)
        proj_dim = int(proj_dim)
        if is_random:
            self.linear_1 = _TransLinear(
                seq_dim, out_len, proj_dim, is_preact, None)
        else:
            self.linear_1 = _TransLinear(
                seq_dim, out_len, proj_dim, is_preact, norm_layer)
        self.add_module('linear_1', self.linear_1)
        self.linear_2 = _TransLinear(
            proj_dim, out_len, seq_dim, is_preact, norm_layer)
        self.add_module('linear_2', self.linear_2)

        if is_random:
            for parameter in self.linear_1.parameters():
                parameter.requires_grad = False

        self.is_random = is_random
        self.is_skip = is_skip
        self.is_preact = is_preact
        self.is_first_relu = is_first_relu

    def forward(self, x):
        # x = [batch_size, in_dim, x_len, ]
        # y = [batch_size, in_dim, y_len, ]
        is_skip = self.is_skip
        is_preact = self.is_preact
        is_first_relu = self.is_first_relu

        if is_preact:
            if is_skip:
                h = self.normalize(x)
                if is_first_relu:
                    h = self.relu_0(h)
            else:
                h = x
            h = self.linear_0(h)
        else:
            h = self.normalize(x)
            h = self.linear_0(h)
        if is_skip:
            x = x + h
        else:
            x = h
        if not is_preact:
            x = self.relu_0(x)

        is_random = self.is_random
        if is_random:
            for parameter in self.linear_1.parameters():
                parameter.requires_grad = False
        h = self.linear_1(x)
        x = x + self.linear_2(h)
        if not is_preact:
            x = self.relu_0(x)
        return x


class RPMixer(nn.Module):
    """
    RPMixer network
    """

    def __init__(self, seq_len, pred_len, seq_dim, feat_dim,
                 proj_dim, dim_factor, n_layer, norm_layer,
                 is_preact, is_random, is_normal, is_freq):
        super(RPMixer, self).__init__()
        self.model_name = 'RPMixer'

        seq_feat_len = seq_len + feat_dim

        if proj_dim == -1:
            proj_dim = np.sqrt(seq_dim)
            proj_dim = int(np.ceil(proj_dim))

        layers = OrderedDict()
        for i in range(n_layer):
            is_skip = True
            if i == 0:
                is_first_relu = False
            else:
                is_first_relu = True
            layers[f'block_{i}'] = _Block_1(
                seq_feat_len, seq_feat_len, seq_dim, proj_dim, dim_factor,
                norm_layer, is_preact, is_random, is_freq,
                is_skip, is_first_relu)

        encoder = nn.Sequential(layers)
        self.add_module(f'encoder', encoder)
        self.encoder_layers = layers

        decoder = nn.Linear(seq_feat_len, pred_len)
        self.add_module(f'decoder', decoder)
        self.decoder = decoder

        self.n_layer = n_layer
        self.encoder = encoder
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seq_dim = seq_dim
        self.is_normal = is_normal

    def forward(self, x):
        # x = [batch_size, seq_dim, seq_len, ]
        # y = [batch_size, seq_dim, pred_len, ]
        is_normal = self.is_normal
        seq_len = self.seq_len
        if is_normal:
            x_ = x.detach()
            x_mu = torch.mean(x_[:, :, :seq_len], 2, keepdim=True)
            x_sigma = torch.std(x_[:, :, :seq_len], 2, keepdim=True)
            x_sigma[x_sigma < 1e-6] = 1.0
            x[:, :, :seq_len] = (x[:, :, :seq_len] - x_mu) / x_sigma

        h = self.encoder(x)
        y = self.decoder(h)

        if is_normal:
            y = (y * x_sigma) + x_mu
        return y

    def forward_i(self, x, i):
        # x = [batch_size, seq_dim, seq_len, ]
        # y = [batch_size, seq_dim, pred_len, ]
        is_normal = self.is_normal
        seq_len = self.seq_len
        if is_normal:
            x_ = x.detach()
            x_mu = torch.mean(x_[:, :, :seq_len], 2, keepdim=True)
            x_sigma = torch.std(x_[:, :, :seq_len], 2, keepdim=True)
            x_sigma[x_sigma < 1e-6] = 1.0
            x[:, :, :seq_len] = (x[:, :, :seq_len] - x_mu) / x_sigma

        h = self.encoder_layers[f'block_{i}'](x)
        y = self.decoder(h)

        if is_normal:
            y = (y * x_sigma) + x_mu
        return y

    def get_n_param(self):
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param

