# -*- encoding: utf-8 -*-
'''
@File    :   learner.py
@Time    :   2024/11/10 10:53:34
@Author  :   junewluo 
'''

import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from a1_net import MLPBaseNet
from utils import check, CIFAR10, get_optimizer, which_loss_criterion, get_activate_func
from sklearn.metrics import confusion_matrix
from loguru import logger

class CifarClassification(object):
    def __init__(self, args):
        self._args = args
        self._epochs = args.epochs
        self._lr = args.lr
        self._use_lr_deacy = args.use_lr_decay
        self._num_classes = args.num_classes
        self._device = args.device
        self._valid_interval = args.valid_interval
        self._classes = args.classes
        self._log_interval = args.log_interval
        self._args.activate_func = get_activate_func(args.activate_func)
        

        self.net = MLPBaseNet(args = self._args).to(self._device)
        self.optimizer = get_optimizer(optim_name = self._args.optim, net = self.net, lr = args.lr, momentum = args.momentum)
        self.criterion = which_loss_criterion(loss_name = self._args.loss)

        self.train_loader = None
        self.valid_loader = None
        self.eval_loader = None

    def linear_lr_decay(self, cur_steps, total_steps):
        lr_now = max(1e-5, self._lr * (1 - cur_steps / total_steps))
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        self._lr = lr_now
        
    
    def load_loader(self, train_loader, valid_loader, eval_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader

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
        log_info = {
            "train_loss": [],
            "train_acc" : [],
            "valid_loss" : [],
            "valid_acc": [],
        }
        img_info = {}

        logger.info(f"training will start! total epoch is {self._epochs}, the size of training data is {len(self.train_loader)} batch and size of validation data is {len(self.valid_loader)} batch")
        valid_steps = 0
        for i in range(self._epochs):
            mean_train_loss , train_acc = self.one_episode(self.train_loader)
            log_info["train_acc"].append(train_acc)
            log_info["train_loss"].append(mean_train_loss)

            if i % self._log_interval == 0:
                logger.info(f"train steps: {i}  |  train acc: {round(train_acc,5) * 100}%  |  train loss: {round(mean_train_loss, 8)}")

            if i % self._valid_interval == 0 or (i+1) == self._epochs:
                mean_valid_loss, valid_acc = self.one_episode(self.valid_loader, ban_grad = True)
                log_info["valid_acc"].append(valid_acc)
                log_info["valid_loss"].append(mean_valid_loss)
                logger.success(f"validation execute success! valid steps: {valid_steps}  |  valid acc: {round(train_acc,5) * 100}%  |  valid loss: {round(mean_train_loss, 3)}")
                valid_steps += 1

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
        fig_name = f"{self._args.optim}_{self._args.loss}_{self._args.batch_size}.png"
        plt.savefig(os.path.join("./result/", fig_name))

        test_acc = np.sum(y_pred == y_true) / y_pred.shape[0]

        return test_acc ,fig_name




