# -*- encoding: utf-8 -*-
'''
@File    :   a2_learner.py
@Time    :   2024/11/14 16:09:39
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
from utils import CNNBase, which_loss_criterion, get_optimizer, check
from loguru import logger

class A2_Learner(BaseLearner):
    def __init__(self, 
                 args,
                ):
        super(A2_Learner, self).__init__(args)

        self._num_classes = args.num_classes
        self._classes = args.classes
        self._input_size = args.input_size
        self._channels = args.channels
        self._kernels = args.kernels
        self._paddings = args.paddings
        self._strides = args.strides
        self._pool_method = args.pool_method
        self._pool_size = args.pool_kernel_size
        self._hidden_dims = args.hidden_dims

        self._use_laryer_norm = args.use_layer_norm
        self._laryer_norm_method = args.layer_norm_method
        self._use_dropout = args.use_dropout
        self._dropout_ratio = args.dropout_ratio

        self._use_l1_norm = args.use_l1_norm
        self._l1_norm_lambda = args.l1_norm_lambda
        self._use_l2_norm = args.use_l2_norm
        self._l2_norm_lambda = args.l2_norm_lambda

        self.net = CNNBase(input_size = args.input_size,
                           channels = self._channels,
                           kernels = self._kernels,
                           paddings = self._paddings,
                           strides = self._strides,
                           pool_method = self._pool_method,
                           pool_size = self._pool_size,
                           activate_func = self._activate_func,
                           hidden_dims = self._hidden_dims,
                           num_classes = self._num_classes,
                           use_layer_norm = self._use_laryer_norm,
                           layer_norm_method = self._laryer_norm_method,
                           use_dropout = self._use_dropout,
                           dropout_ratio = self._dropout_ratio,
                        ).to(self._device)
        logger.success(self.net)
        self.optimizer = get_optimizer(optim_name = args.optim, net = self.net, lr = args.lr, momentum = args.momentum)
        self.criterion = which_loss_criterion(loss_name = self._loss)

    def linear_lr_decay(self):
        return super().linear_lr_decay()
    
    def make_confuse_matrix(self, y_true, y_pred, fig_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6),dpi=120)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self._classes, yticklabels=self._classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join("./result/", fig_name))

        return fig_name

    def update(self, dataloader):
        loss_, correct_label = 0.0, 0
        for _, (x, y) in enumerate(dataloader):
            x = check(x).to(self._device)
            one_hot_y = F.one_hot(y, num_classes = self._num_classes).float().to(self._device)
            probs = self.net(x)

            if self._use_l1_norm:
                regular = torch.sum(torch.sum(torch.abs(param)) for param in self.net.parameters()) * self._l1_norm_lambda
            elif self._use_l2_norm:
                regular = torch.sum(p.pow(2.0).sum() for p in self.net.parameters()) * self._l2_norm_lambda
            else:
                regular = 0.0

            loss = self.criterion(probs, one_hot_y).mean() + regular
            # update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            # store metrics
            pred_label = torch.argmax(probs.detach().cpu(), dim = 1).view(-1)
            if len(pred_label.shape) != len(y.shape):
                y = y.view(-1)
            correct_label += torch.sum((pred_label == y)).item()
            loss_ += loss.detach().cpu().item()
        
        return loss_ / len(dataloader), correct_label / len(dataloader.dataset)
    
    @torch.no_grad()
    def eval(self, dataloader, make_confuse = False, fig_name = None):
        predict_labels = []
        true_labels = []
        eval_loss = 0.0

        for _, (x, y) in enumerate(dataloader):
            x = check(x).to(self._device)
            one_hot_y = F.one_hot(y, num_classes = self._num_classes).float().to(self._device)
            probs = self.net(x)
            eval_loss += self.criterion(probs, one_hot_y).mean().cpu().numpy().item()
            pred_label = torch.argmax(probs, dim = 1).view(-1).cpu().numpy()
            y = y.view(-1).numpy()

            predict_labels.extend(pred_label.tolist())
            true_labels.extend(y.tolist())

        y_pred = np.array(predict_labels)
        y_true = np.array(true_labels)
        acc = np.sum(y_pred == y_true) / y_pred.shape[0]

        if make_confuse:
            fig_name = self.make_confuse_matrix(y_true = true_labels, y_pred = predict_labels, fig_name = fig_name)
        else:
            fig_name = None
        
        return fig_name, acc, eval_loss / len(dataloader)

    def learn(self, train_loader, valid_loader, eval_loader):
        self.load_loader(train_loader, valid_loader, eval_loader)

        start_time = self.cur_timestamp()
        for i in range(self._epochs):

            log_info = dict()
            log_info["steps"] = i

            # training 
            train_loss, train_acc = self.update(dataloader = self.train_loader)
            log_info["train_loss"] = train_loss
            log_info["train_acc"] = train_acc

            # validation
            if i % self._valid_interval == 0 or (i+1) == self._epochs:
                _, valid_acc, valid_loss = self.eval(self.valid_loader)
                log_info["valid_acc"]  = valid_acc
                log_info["valid_loss"] = valid_loss
                logger.success(f'validation execute success! valid steps: {i//self._valid_interval}  |  valid acc: {round(valid_acc,5) * 100}%  |  valid loss: {round(valid_loss, 3)}')
                
            
            if i % self._log_interval == 0:
                self.log_info(log_info)
        
        
        end_time = self.cur_timestamp()
        hours, minitues, seconds = self.run_time(start_time, end_time)
        logger.success(f"fininsh training successfully! current time is {self.cur_timestamp(True)}, training process takes a total of {int(hours)}:{int(minitues)}:{int(seconds)}")

                