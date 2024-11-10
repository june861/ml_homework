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
                 activate_func = F.relu,
                 hidden_dims = [],
                ):
        
        super(MLPLayers, self).__init__()
        input_layers_dims = [input_dim] + hidden_dims[:-1]
        output_layers_dims = hidden_dims
        self.activate_func = activate_func
        self.layers = []

        for index, (i_d, o_d) in enumerate(zip(input_layers_dims, output_layers_dims)):
            fc = nn.Linear(i_d, o_d)
            self.layers.append(fc)
        self.layers = nn.ModuleList(self.layers)
        MLPLogger.info(f"net has been init successfully! input_dim is {input_dim}, output_dim is {hidden_dims[-1]}, net has {len(self.layers)} layers!")


    def forward(self, x, last_activate = True):
        for fc in self.layers[:-1]:
            x = self.activate_func(fc(x))
        if last_activate:
            return self.activate_func(self.layers[-1](x))
        return self.layers[-1](x)
    







