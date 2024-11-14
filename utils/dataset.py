# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2024/11/10 10:19:06
@Author  :   junewluo 
'''

import random
import torch
import torchvision
from torch.utils.data import random_split
from loguru import logger


class CIFAR10(object):
    def __init__(self,save_path = "./data/" ,
                 LOAD_CIFAR = True, 
                 DOWNLOAD_CIFAR = True, 
                 transform_train = None, 
                 transforma_eval = None,
                 split_ratio = 0.0
                ):
        self.load_cifar = LOAD_CIFAR
        self.download_cifar = DOWNLOAD_CIFAR
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
                    )
        self.save_path = save_path
        self.transform_train = transform_train
        self.transform_eval = transforma_eval
        self.split_ratio = split_ratio

        self.train_data = None
        self.test_data = None
        self.valid_data = None
    
    def load_cifar_from_torch(self, load_train = True):
        tmp = torchvision.datasets.CIFAR10(
                                            root = self.save_path ,
                                            train = load_train,
                                            transform = self.transform_train if load_train else self.transform_eval,
                                            download = self.download_cifar,
                                        )
        if load_train:
            self.train_data = tmp
        else:
            self.test_data = tmp
    

    def random_split_data(self):
        valid_length = int(len(self.train_data) * self.split_ratio)
        train_length = len(self.train_data) - valid_length 
        train_dataset, valid_dataset = random_split(dataset = self.train_data, 
                                                lengths= [train_length, valid_length], 
                                                generator = torch.Generator().manual_seed(random.randint(0,100))
                                                )
        
        self.train_data = train_dataset
        self.valid_data = valid_dataset
        logger.info(f"split validation dataset from train dataset! split ratio is {self.split_ratio}")
