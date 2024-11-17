# -*- encoding: utf-8 -*-
'''
@File    :   learner.py
@Time    :   2024/11/10 10:53:34
@Author  :   junewluo 
'''

import os
import wandb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from utils import MLPBase
from utils import check, CIFAR10, get_optimizer, which_loss_criterion
from sklearn.metrics import confusion_matrix
from loguru import logger
from .base_learners import BaseLearner

class A1_Learner(BaseLearner):
    def __init__(self, args):
        super(A1_Learner, self).__init__(args)

        self._num_classes = args.num_classes
        self._classes = args.classes

        self.net = MLPBase(input_dim = args.input_dim, 
                           activate_func = self._activate_func,
                           hidden_dims = args.hidden_dims,
                           num_classes = self._num_classes,
                        ).to(self._device)
        self.optimizer = get_optimizer(optim_name = args.optim, net = self.net, lr = args.lr, momentum = args.momentum)
        self.criterion = which_loss_criterion(loss_name = args.loss)


    def linear_lr_decay(self, cur_steps, total_steps):
        lr_now = max(1e-5, self._lr * (1 - cur_steps / total_steps))
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        self._lr = lr_now


    def one_episode(self, dataloader, ban_grad = False):
        loss_, correct_label = 0.0, 0
        for _, (x, y) in enumerate(dataloader):
            x = check(x).to(self._device)
            one_hot_y = F.one_hot(y, num_classes = self._num_classes).float().to(self._device)
            if ban_grad == False:
                probs = self.net(x)
                loss = self.criterion(probs, one_hot_y).mean()
                # update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    probs = self.net(x)
                    loss = self.criterion(probs, one_hot_y).mean()

            # store metrics
            pred_label = torch.argmax(probs.detach().cpu(), dim = 1).view(-1)
            if len(pred_label.shape) != len(y.shape):
                y = y.view(-1)
            correct_label += torch.sum((pred_label == y)).item()
            loss_ += loss.detach().cpu().item()
        
        return loss_ / len(dataloader), correct_label / len(dataloader.dataset)


    def learn(self, train_loader, valid_loader, eval_loader):
        self.load_loader(train_loader, valid_loader, eval_loader)
        img_info = {}

        logger.info(f"training will start! total epoch is {self._epochs}, the size of training data is {len(self.train_loader)} batch and size of validation data is {len(self.valid_loader)} batch")
        valid_steps = 0
        for i in range(self._epochs):
            log_info = {}
            log_info["steps"] = i

            mean_train_loss , train_acc = self.one_episode(self.train_loader)
            log_info["train_acc"] = train_acc
            log_info["train_loss"] = mean_train_loss

            if i % self._valid_interval == 0 or (i+1) == self._epochs:
                mean_valid_loss, valid_acc = self.one_episode(self.valid_loader, ban_grad = True)
                log_info["valid_acc"] = valid_acc
                log_info["valid_loss"]=  mean_valid_loss
                logger.success(f"validation execute success! valid steps: {valid_steps}  |  valid acc: {round(valid_acc,5) * 100}%  |  valid loss: {round(mean_valid_loss, 5)}")
                valid_steps += 1

            if i % self._log_interval == 0:
                # logger.info(f"train steps: {i}  |  train acc: {round(train_acc,5) * 100}%  |  train loss: {round(mean_train_loss, 8)}")
                self.log_info(log_info)

            if self._use_lr_deacy:
                self.linear_lr_decay(cur_steps = i, total_steps = self._epochs)
            


        test_acc, fig_name = self.eval()
        img_info["fig_name"] = fig_name
        logger.success(f"training fininsh! testing fininsh! a confusion matrix has been saved to {fig_name}. you can check it! test acc is {test_acc}")

        return log_info, img_info


    @torch.no_grad()
    def eval(self):
        predict_labels = []
        true_labels = []

        for _, (x, y) in enumerate(self.eval_loader):
            x = check(x).to(self._device)
            probs = self.net(x)

            pred_label = torch.argmax(probs, dim = 1).view(-1).cpu().numpy()
            y = y.view(-1).numpy()

            predict_labels.extend(pred_label.tolist())
            true_labels.extend(y.tolist())
        

        y_pred = np.array(predict_labels)
        y_true = np.array(true_labels)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6),dpi=120)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self._classes, yticklabels=self._classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.title('Confusion Matrix')
        fig_name = f"{self._optim}_{self._loss}_{self._batch_size}.png"
        plt.savefig(os.path.join("./result/", fig_name))

        test_acc = np.sum(y_pred == y_true) / y_pred.shape[0]

        return test_acc ,fig_name




