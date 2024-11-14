# -*- encoding: utf-8 -*-
'''
@File    :   mlp.py
@Time    :   2024/11/10 09:52:10
@Author  :   junewluo 
'''

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as MLPLogger


class MLPLayers(nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 activate_func = nn.ReLU(),
                 hidden_dims = [],
                 use_laryer_norm = False,
                 layer_norm_method = None,
                 use_dropout = False,
                 dropout_ratio = 0.2,
                ):
        
        super(MLPLayers, self).__init__()
        input_layers_dims = [input_dim] + hidden_dims[:-1]
        output_layers_dims = hidden_dims
        self.activate_func = activate_func
        self.layers = []

        for index, (i_d, o_d) in enumerate(zip(input_layers_dims, output_layers_dims)):
            fc = nn.Linear(i_d, o_d)
            self.layers.append(fc)
            self.layers.append(self.activate_func)
            if use_laryer_norm:
                self.layers.append(nn.BatchNorm1d(o_d) if layer_norm_method == "batch_norm" else nn.LazyBatchNorm1d(o_d))
            elif use_dropout:
                self.layers.append(nn.Dropout(dropout_ratio))
        self.layers = nn.ModuleList(self.layers)
        MLPLogger.info(f"net has been init successfully! input_dim is {input_dim}, output_dim is {hidden_dims[-1]}, net has {len(self.layers)} layers!")


    def forward(self, x, last_activate = True):
        for layer in self.layers:
            x = layer(x)
        return x
    
class MLPBase(nn.Module):
    def __init__(self, 
                 input_dim, 
                 activate_func, 
                 hidden_dims, 
                 num_classes,
                 use_laryer_norm = False,
                 layer_norm_method = None,
                 use_dropout = False,
                 dropout_ratio = 0.2,
                ):
        super(MLPBase, self).__init__()
        
        self.ahead_net = MLPLayers(input_dim = input_dim, 
                                   activate_func = activate_func,
                                   hidden_dims = hidden_dims,
                                   use_laryer_norm = use_laryer_norm,
                                   layer_norm_method = layer_norm_method,
                                   use_dropout = use_dropout,
                                   dropout_ratio = dropout_ratio,
                                )
        self.classes_layers = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x, use_softmax = True):
        hidden_out = self.ahead_net(x)
        if use_softmax:
            return F.softmax(self.classes_layers(hidden_out))
        return hidden_out



