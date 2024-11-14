# -*- encoding: utf-8 -*-
'''
@File    :   base_learners.py
@Time    :   2024/11/14 16:00:32
@Author  :   junewluo 
'''
import wandb
import os
import time
import datetime
from utils import get_activate_func
from loguru import logger

class BaseLearner(object):
    def __init__(self, args):

        self._ppid = os.getppid()
        self._pid = os.getpid()
        self._date = datetime.datetime.now().strftime(r"%Y-%m-%d %H-%M-%S") 
        self._task = args.task
        self._batch_size = args.batch_size
        self._optim = args.optim
        self._loss = args.loss
        self._lr = args.lr
        self._epochs = args.epochs
        self._device = args.device
        self._valid_interval = args.valid_interval
        self._use_lr_deacy = args.use_lr_decay
        self._log_interval = args.log_interval
        self._activate_func = get_activate_func(args.activate_func)
        self._use_wandb = args.use_wandb
        if not self._use_wandb:
            self._tb_writer = args.tb_writer

        self.train_loader = None
        self.valid_loader = None
        self.eval_loader = None

    def load_loader(self, train_loader, valid_loader, eval_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.eval_loader = eval_loader

    def linear_lr_decay(self):
        raise NotImplementedError

    def uptdae(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError(f'method learn has been not yet implement')

    def eval(self):
        raise NotImplementedError

    def confuse_matrix(self):
        raise NotImplementedError
    
    def cur_timestamp(self, formate_date = False):
        if formate_date:
            return datetime.datetime.now().strftime(r"%Y-%M-%D %H:%M:%S")
        return time.time()

    def run_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        # convert to hours, minitute, second
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        return hours, minutes, seconds

    def log_info(self, info):
        logger.info(f"train steps: {info['steps']}  |  train acc: {round(info['train_acc'], 5) * 100}%  |  train loss: {round(info['train_loss'], 8)}")
        if self._use_wandb:
            wandb.log(info)
        else:
            for k,v in info.items():
                self._tb_writer.add_scalar(k,v)
