# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/11/10 10:02:53
@Author  :   junewluo 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def check(input):
    return torch.from_numpy(input) if type(input) == np.ndarray else input

class FlattenTo1D(object):
    def __call__(self, input):
        return input.view(-1)


def get_optimizer(optim_name, net: nn.Module, lr: float, **kwargs):
    if optim_name == "adam":
        optimizer = optim.Adam(params = net.parameters(), lr = lr)
    elif optim_name == "adamw":
        optimizer = optim.AdamW(params = net.parameters(), lr = lr)
    elif optim_name == "adagrad":
        optimizer = optim.Adagrad(params = net.parameters(), lr = lr)
    elif optim_name == "sgd":
        optimizer = optim.SGD(params = net.parameters(), lr = lr)
    elif optim_name == "momentum":
        optimizer = optim.SGD(params = net.parameters(), lr = lr, momentum = kwargs["momentum"])
    else:
        raise NotImplementedError()
    return optimizer


def which_loss_criterion(loss_name):
    if loss_name == "mse":
        return F.mse_loss
    elif loss_name == "entropy":
        return F.cross_entropy
    else:
        raise NotImplementedError()


def get_activate_func(activate_func_name):
    if activate_func_name == "relu":
        return F.relu
    elif activate_func_name == "sigmoid":
        return F.sigmoid
    elif activate_func_name == "tanh":
        return F.tanh