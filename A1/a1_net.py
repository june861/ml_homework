# -*- encoding: utf-8 -*-
'''
@File    :   a1_net.py
@Time    :   2024/11/10 10:31:30
@Author  :   junewluo 
'''

import torch.nn as nn
import torch.nn.functional as F
from utils import MLPLayers

class MLPBaseNet(nn.Module):
    def __init__(self, args):
        super(MLPBaseNet, self).__init__()
        
        self.ahead_net = MLPLayers(input_dim = args.input_dim, 
                                   activate_func = args.activate_func,
                                   hidden_dims = args.hidden_dims,
                                )
        self.classes_layers = nn.Linear(args.hidden_dims[-1], args.num_classes)
    
    def forward(self, x):
        hidden_out = self.ahead_net(x)
        softmax_out = F.softmax(self.classes_layers(hidden_out))
        return softmax_out
