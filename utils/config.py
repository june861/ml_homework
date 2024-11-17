# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2024/11/10 10:40:19
@Author  :   junewluo 
'''

import argparse
from loguru import logger

def get_config():
    parser = argparse.ArgumentParser(
        description='ml_homework', formatter_class=argparse.RawDescriptionHelpFormatter)
    # task description
    parser.add_argument("--task", type=str, help="task name",required=True)

    # cuda device setting
    parser.add_argument("--use_cuda", action="store_false", default=True, help="whether use cuda device to execute training process")
    parser.add_argument("--cuda_rank", type=int, default=0, help="which cuda device to use")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="use wandb or tensorboard")
    parser.add_argument("--group", type=str, default=None, help="group name of wandb project")
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    # Network Setting
    parser.add_argument("--use_layer_norm", action="store_true", default=False, help="use a normalization method")
    parser.add_argument("--layer_norm_method", type=str, default="batch_norm", help="which layer norm method is used", choices=["layer_norm", "batch_norm"])
    parser.add_argument("--use_dropout", action="store_true", default=False, help="whether use to dropout method")
    parser.add_argument("--dropout_ratio", type=float, default=0.2, help="dropout ratio")

    parser.add_argument("--log_interval", type=int, default=1, help="log metric interval")
    parser.add_argument("--optim", type=str, default="adam", help="which optimizer should be use to update grad")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size of a batch data")
    parser.add_argument("--activate_func", type=str, default="relu", help="the activate function for linear output")
    parser.add_argument("--hidden_dims", nargs="+", default=[128,128], help="the dimensions of hidden layers")
    parser.add_argument("--epochs", type=int, default=100, help="the total training epoch")
    parser.add_argument("--valid_interval", type=int, default=3, help="the validation interval when training")
    parser.add_argument("--lr", type=float, default=1e-3, help="update steps")
    
    parser.add_argument("--use_lr_decay", action="store_false", default=True, help="using a trick: learning rate decay")
    parser.add_argument("--lr_decay_policy", type=str, default="linear", help="which poilcy will be selected")

    return parser

def parse_args(args, parser):
    logger.success(f"test decorator!")