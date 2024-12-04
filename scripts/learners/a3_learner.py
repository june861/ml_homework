# -*- encoding: utf-8 -*-
'''
@File    :   a3_learner.py
@Time    :   2024/11/29 18:13:54
@Author  :   junewluo 
'''

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import confusion_matrix
from scripts.learners.base_learners import BaseLearner
from utils import RNNBase, get_optimizer, which_loss_criterion
from loguru import logger
from torch.utils.data import Dataset


class A3Learner(BaseLearner):
    def __init__(self, all_args, vocab_size, len_name):
        super(A3Learner, self).__init__(args = all_args)
        
        self._embed_dim = all_args.embed_dim
        self._hidden_size = all_args.hidden_size
        self._batch_first = all_args.batch_first
        self._bidirectional = all_args.bidirectional
        self._num_layers = all_args.num_layers
        self._bidirectional = all_args.bidirectional
        self._use_log_softmax = True if self._loss == "nll" else False
        self._name_length = len_name
        

        self.rnn_net = RNNBase(vocab_size = vocab_size, 
                               embed_dim = self._embed_dim,
                               hidden_size = self._hidden_size,
                               num_layers = self._num_layers,
                               batch_first = self._batch_first,
                               output_size = vocab_size,
                               activate_func = self._activate_func,
                               bidirectional = self._bidirectional,
                            ).to(self._device)


        self.optimizer = get_optimizer(optim_name = self._optim, net = self.rnn_net, lr = self._lr)
        self.criterion = which_loss_criterion(loss_name = self._loss)

    def get_net_parameter(self):
        return self.rnn_net.parameters()

    def set_share_model(self):
        self.rnn_net.share_memory()

    def forward(self, x):
        return self.rnn_net(x, self._use_log_softmax).float()
        
    def cal_loss(self, pred_y, true_y):
        if self._loss == "nll":
            true_y = torch.argmax(true_y, dim=1)
        else:
            true_y = true_y.float()
        
        return self.criterion(pred_y, true_y)

    def linear_lr_decay(self, cur_steps, total_steps):
        lr_now = max(1e-5, self._lr * (1 - cur_steps / total_steps))
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        self._lr = lr_now
    
    def update(self):
        loss_ = 0.0
        loss = 0.0
        for idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self._device)
            y = y.to(self._device)
            # x.shape is (batch_size, self._name_length, vocab_size)
            # y.shape is (batch_size, self._name_length, vocab_size)
            
            for name_index in range(self._name_length):
                x_ = x[:,name_index,:]
                y_ = y[:,name_index,:]
                
                if self._loss == "nll":
                    rnn_out = self.forward(x_)
                    indice = torch.argmax(y_, dim = 1)
                    loss += self.criterion(rnn_out, indice)
                else:
                    rnn_out = self.forward(x_)
                    loss += self.criterion(rnn_out, y_.float())

        self.optimizer.zero_grad()
        # loss /= x.shape[0]
        loss.backward()
        self.optimizer.step()

        loss_ += loss.detach().cpu().numpy().item()

        return loss_ / len(self.train_loader)
    
    
    @torch.no_grad()
    def eval(self, validation = False):
        if validation:
            dataloader = self.valid_loader
        else:
            dataloader = self.eval_loader
    
    def log_info(self, info):
        logger.info(f"train steps: {info['steps']}  |  train loss: {round(info['train_loss'], 8)}")
        if self._use_wandb:
            wandb.log(info)
        else:
            for k,v in info.items():
                self._tb_writer.add_scalar(k,v)


    def learn(self, trainloader):
        self.train_loader = trainloader
        
        for i in range(self._epochs):
            infos = {"steps":i}

            train_loss = self.update()
            infos["train_loss"] = train_loss 
            
            
            if i % self._log_interval == 0:
                self.log_info(info = infos)
            
            if i % self._valid_interval == 0:
                self.linear_lr_decay(i, self._epochs)
            