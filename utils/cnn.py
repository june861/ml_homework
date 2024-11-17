# -*- encoding: utf-8 -*-
'''
@File    :   cnn.py
@Time    :   2024/11/10 21:56:35
@Author  :   junewluo 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from .mlp import MLPBase
from utils import get_pool_func, get_activate_func
from loguru import logger

class ConvLayers(nn.Module):
    def __init__(self, 
                 input_size : tuple,
                 channels : list, 
                 kernels : list,
                 paddings : list,
                 strides : list,
                 pool_kernel : tuple,
                 pool_method = None,
                ):
        super(ConvLayers, self).__init__()

        if channels[0] != input_size[0]:
            raise ValueError(f'channels[0] must be equal to input_size[0]. ')

        self.channel = input_size[0]
        self.w = input_size[1]
        self.h = input_size[2]
        in_channels = channels[:-1]
        out_channels = channels[1:]

        self.pools = []
        self.convs = []

        pool_padding = 0
        for index, (i_c, o_c, k, p, s) in enumerate(zip(in_channels, out_channels, kernels, paddings, strides)):
            conv_layer = nn.Conv2d(in_channels = i_c, out_channels = o_c, kernel_size = k, padding = p, stride = s)
            self.channel = o_c
            # after convolution layers
            self.w = floor((self.w + 2 * p - k[0]) / s) + 1
            self.h = floor((self.h + 2 * p - k[1]) / s) + 1
            # after pooling layers
            pool_stride_w = pool_kernel[index][0]
            pool_stride_h = pool_kernel[index][1]
            self.w = floor((self.w + 2 * pool_padding - pool_kernel[index][0]) / pool_stride_w) + 1
            self.h = floor((self.h + 2 * pool_padding - pool_kernel[index][1]) / pool_stride_h) + 1
            self.convs.append(conv_layer)
            pool = get_pool_func(pool_method, pool_kernel[index])
            logger.info(f'use pool of {type(pool)}, pool metho is {pool_method}, current layer is {index}')
            self.pools.append(pool)
        
        self.convs = nn.ModuleList(self.convs)

    def forward(self, x, flatten = True):
        """ convolutional network forward

        Args:
            x (_tensor_): input tensor
            flatten (bool): whether to flatten a tensor shape into (batch, -1). Defaults to True.

        Returns:
            _type_: _description_
        """
        index = 0
        for conv_layer in self.convs:
            x = self.pools[index](conv_layer(x))
            index += 1
        if flatten:
            x = torch.reshape(x, shape=(x.shape[0], -1))
        return x



class CNNBase(nn.Module):
    def __init__(self, 
                 input_size, 
                 channels, 
                 kernels, 
                 paddings, 
                 strides,
                 pool_method,
                 pool_size,
                 activate_func,
                 hidden_dims,
                 num_classes,
                 use_layer_norm,
                 layer_norm_method,
                 use_dropout,
                 dropout_ratio,
                ):
        super(CNNBase, self).__init__()

        self.conv_net = ConvLayers(input_size = input_size, 
                                   channels = channels, 
                                   kernels = kernels, 
                                   paddings = paddings,
                                   strides = strides,
                                   pool_kernel = pool_size,
                                   pool_method = pool_method,
                                )
        self.mlp_input_dim = self.conv_net.channel * self.conv_net.w * self.conv_net.h
        self.mlp_net = MLPBase(input_dim = self.mlp_input_dim, 
                               activate_func = activate_func,
                               hidden_dims = hidden_dims,
                               num_classes = num_classes,
                               use_laryer_norm = use_layer_norm,
                               layer_norm_method = layer_norm_method,
                               use_dropout = use_dropout,
                               dropout_ratio = dropout_ratio,
                            )


    def forward(self, x):
        conv_out = self.conv_net(x)
        if len(conv_out.shape) > 2:
            conv_out = conv_out.reshape(conv_out.shape[0], -1)
        mlp_out = self.mlp_net(conv_out)
        return mlp_out
            

