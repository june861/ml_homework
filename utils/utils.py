# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/11/10 10:02:53
@Author  :   junewluo 
'''

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from loguru import logger
from sklearn.model_selection import KFold


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
    elif optim_name == "admax":
        optimizer = optim.Adamax(params = net.parameters(), lr = lr)
    elif optim_name == "rmsprop":
        optimizer = optim.RMSprop(params = net.parameters(), lr = lr)
    else:
        logger.error(f"we only support [adam, adamw, adagrad, sgd, momentum, admax, rmsprop] optimizer. but now recieve {optim_name}")
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
        return nn.ReLU()
    elif activate_func_name == "sigmoid":
        return nn.Sigmoid()
    elif activate_func_name == "tanh":
        return nn.Tanh()

def get_pool_func(pool_method, pool_size = (2,2), stride = None, pool_padding = 0):
    if pool_method == "max_pool":
        return nn.MaxPool2d(kernel_size = pool_size, stride = stride, padding = pool_padding)
    elif pool_method == "avg_pool":
        return nn.AvgPool2d(kernel_size = pool_size, stride = stride, padding = pool_padding)
    else:
        raise NotImplementedError

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Kfold_Dataset(kfolds = 5, shuffle = True):
    return KFold(n_splits = kfolds, shuffle = shuffle)


def regular_normalization_name(args):
    if args.use_l1_norm:
        regular = "L1"
    elif args.use_l2_norm:
        regular = "L2"
    else:
        regular = "null"
    
    if args.use_layer_norm:
        normalization = f"layernorm{args.layer_norm_method}"
    elif args.use_dropout:
        normalization = f"dropout({args.dropout_ratio})"
    else:
        normalization = "null"
    
    return regular, normalization