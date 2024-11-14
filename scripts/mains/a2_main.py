# -*- encoding: utf-8 -*-
'''
@File    :   a2_runner.py
@Time    :   2024/11/14 16:08:22
@Author  :   junewluo 
'''
import argparse
import os
import time
import torch
import wandb
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scripts.learners.a2_learner import A2_Learner
from utils import (
    get_config, parse_args, FlattenTo1D, 
    CIFAR10, set_seed, Kfold_Dataset, 
    regular_normalization_name,
)
from torchvision.transforms import (
    ToTensor, Normalize, transforms,
    )
from torch.utils.data import (
    DataLoader, SubsetRandomSampler,
)
from tensorboardX import SummaryWriter
from loguru import logger

def parse_args(args, parser: argparse.ArgumentParser):

    parser.add_argument("--split_ratio", type=float, default=0.3, help="random split ratio for spliting training datatset")
    parser.add_argument("--loss", type=str, default="entropy", help="which loss function will be used", choices=["mse","entropy"])
    parser.add_argument("--momentum", type=float, default=0.2, help="momentum optimizer")
    
    parser.add_argument("--channels", nargs="+", default=[64,128,256], help="convolutional layers channels")
    parser.add_argument("--kernels", nargs="+", default=[(3,3),(1,1),(1,1)], help="convolutional kernel size")
    parser.add_argument("--paddings", nargs="+", default=[0,0,0], help="padding of convolutional layers")
    parser.add_argument("--strides", nargs="+", default=[1,1,1], help="strides for convolutional kernel")
    parser.add_argument("--pool_method", type=str, default="max_pool", help="method of pooling operation", choices=["max_pool","avg_pool"])
    parser.add_argument("--pool_kernel_size", nargs="+", default=[(2,2),(2,2),(2,2)], help="kernel size of pooling")
    parser.add_argument("--num_classes", type=int, default=10, help="the number of classes when doing classification")
    parser.add_argument("--kfold", type=int, default=5, help="")

    parser.add_argument("--use_l1_norm", action="store_true", default=False, help="")
    parser.add_argument("--l1_norm_lambda", type=float, default=0.01, help="L1-regularization weight")
    parser.add_argument("--use_l2_norm", action="store_true", default=True, help="")
    parser.add_argument("--l2_norm_lambda", type=float, default=0.01, help="L2-regularization weight")


    all_args = parser.parse_known_args(args)[0]

    # 解析特定参数
    all_args.input_dim = 3072
    all_args.input_size = (3,32,32)
    all_args.channels[0] = all_args.input_size[0]
    if all_args.use_cuda and torch.cuda.is_available():
        cuda_rank = all_args.cuda_rank if torch.cuda.device_count() > all_args.cuda_rank else 0
        all_args.device = torch.device(f"cuda:{cuda_rank}")
    else:
        all_args.device = torch.device(f"cpu")
    all_args.classes = [
                        'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
                    ]
    set_seed(all_args.seed)

    return all_args


def main(args):
    # parser arguments from sys
    parser = get_config()
    all_args = parse_args(args, parser)
    transform = transforms.Compose([ToTensor(),
                                    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
                                ])
    transform_eval = transform

    # load dataset
    cifar10_datasets = CIFAR10(transform_train = transform, transforma_eval = transform_eval, split_ratio = 0.3)
    cifar10_datasets.load_cifar_from_torch(load_train = True)
    cifar10_datasets.load_cifar_from_torch(load_train = False)

    
    kfold = Kfold_Dataset(kfolds = all_args.kfold, shuffle = True)
    
    best_test_acc = 0.0
    best_classifier = None
    accs = []
    regular, normalization = regular_normalization_name(all_args)

    for fold, (train_indexs, valid_index) in enumerate(kfold.split(cifar10_datasets.train_data)):
        # wandb init
        if all_args.use_wandb:
            log_name = f'A2_{regular}_{normalization}_{fold}_seed{all_args.seed}_{int(time.time())}'
            wandb.init(
                project = "ml_homework",
                name = log_name,
                group = "A2-v1",
            )
        else:
            log_dir = f'./runs/A2_{regular}_{normalization}_{fold}_seed{all_args.seed}_{int(time.time())}_{os.getppid()}'
            tb_writer = SummaryWriter(log_dir)
            all_args.tb_writer = tb_writer

        train_sampler = SubsetRandomSampler(train_indexs)
        valid_sampler = SubsetRandomSampler(valid_index)

        eval_loader = DataLoader(dataset = cifar10_datasets.test_data, batch_size = all_args.batch_size)
        train_loader = DataLoader(dataset = cifar10_datasets.train_data, sampler = train_sampler, batch_size = all_args.batch_size)
        valid_loader = DataLoader(dataset = cifar10_datasets.train_data, sampler = valid_sampler, batch_size = all_args.batch_size)
        
        cnn_cifar10 = A2_Learner(args = all_args)
        train_acc, valid_acc = cnn_cifar10.learn(train_loader, valid_loader, eval_loader)

        fig_name = f"A2-{regular}_{normalization}_{fold}_{all_args.seed}.png"
        _, test_acc, test_loss = cnn_cifar10.eval(cnn_cifar10.eval_loader, make_confuse = True, fig_name = fig_name)
        
        accs.append((train_acc, valid_acc, test_acc))
        if test_acc > best_test_acc:
            best_classifier = (fold, train_indexs, valid_index)
            best_test_acc = test_acc
    
    acc_dataframe = pd.DataFrame(accs, columns = ["train_acc", "valid_acc", "test_acc"])
    logger.success(f"cross-validation result:\n{acc_dataframe}")
    logger.success(f"cross-validation found best parameters in fold{best_classifier[0]}, best_test_acc is {best_test_acc}")
        


if __name__ == "__main__":
    main(sys.argv[1:])