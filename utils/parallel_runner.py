# -*- encoding: utf-8 -*-
'''
@File       :parallel_runner.py
@Description:
@Date       :2024/12/03 19:38:56
@Author     :junweiluo
@Version    :python
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import time

class ParallelRunner(object):
    def __init__(self):
        pass