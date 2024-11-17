# -*- encoding: utf-8 -*-
'''
@File    :   a1_runner.py
@Time    :   2024/11/14 16:08:08
@Author  :   junewluo 
'''

# -*- encoding: utf-8 -*-
'''
@File    :   a1.py
@Time    :   2024/11/10 10:14:10
@Author  :   junewluo 
'''
 
import os
import time
import torch
import wandb
import sys
import matplotlib.pyplot as plt
from scripts.learners.a1_learner import A1_Learner
from utils import get_config, parse_args, FlattenTo1D, CIFAR10, set_seed
from torchvision.transforms import (
    ToTensor, Normalize, transforms
    )
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

def parse_args(args, parser):

    parser.add_argument("--split_ratio", type=float, default=0.3, help="random split ratio for spliting training datatset")
    #parser.add_argument("--optim", type=str, default="adam", help="which optimizer should be use to update grad")
    parser.add_argument("--loss", type=str, default="entropy", help="which loss function will be used", choices=["mse","entropy"])
    parser.add_argument("--momentum", type=float, default=0.2, help="momentum optimizer")
    #parser.add_argument("--log_interval", type=int, default=1, help="log metric interval")
    parser.add_argument("--num_classes", type=int, default=10, help="the number of classes when doing classification")
    all_args = parser.parse_known_args(args)[0]

    # 解析特定参数
    all_args.input_dim = 3072
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
                                    FlattenTo1D(),
                                ])
    transform_eval = transform



    # wandb init
    if all_args.use_wandb:
        log_name = f'A1_{all_args.optim}_{all_args.loss}_{all_args.batch_size}_{int(time.time())}'
        wandb.init(
            project = "ml_homework",
            name = log_name,
            group = "A1-v1",
        )
    else:
        log_dir = f'./runs/A1_{all_args.optim}_{all_args.loss}_{all_args.batch_size}_seed{all_args.seed}_{int(time.time())}_{os.getppid()}'
        tb_writer = SummaryWriter(log_dir)

    # load dataset
    cifar10_datasets = CIFAR10(transform_train = transform, transforma_eval = transform_eval, split_ratio = 0.3)
    cifar10_datasets.load_cifar_from_torch(load_train = True)
    cifar10_datasets.load_cifar_from_torch(load_train = False)
    cifar10_datasets.random_split_data()

    train_loader = DataLoader(dataset = cifar10_datasets.train_data, batch_size = all_args.batch_size)
    valid_loader = DataLoader(dataset = cifar10_datasets.valid_data, batch_size = all_args.batch_size)
    eval_loader = DataLoader(dataset = cifar10_datasets.test_data, batch_size = all_args.batch_size)

    # define classification instance
    cifar_classification = A1_Learner(args = all_args)

    _, img_info = cifar_classification.learn(train_loader, valid_loader, eval_loader)

    if all_args.use_wandb:
        for fig_name, fig_path in img_info.items():
            wandb.log({str(fig_name) : wandb.Image(os.path.join("./result/", fig_path))})
    else:
        index = 0
        for fig_name, fig_path in img_info.items():
            index += 1
            image = plt.imread(os.path.join("./result/", fig_path))
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            tag = f'{all_args.optim}_{all_args.loss}_{all_args.batch_size}_{index}'
            tb_writer.add_image(tag = tag, img_tensor = img_tensor[:3, :, :])
        

if __name__ == "__main__":
    main(sys.argv[1:])






