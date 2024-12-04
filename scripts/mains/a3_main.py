# -*- encoding: utf-8 -*-
'''
@File    :   a3_main.py
@Time    :   2024/11/29 18:15:18
@Author  :   junewluo 
'''

import os
import sys
import wandb
import torch
import argparse
import time
import torch
import torch.multiprocessing as mp
from collections import Counter
from loguru import logger
from utils import get_config
from tensorboardX import SummaryWriter
from scripts.learners.a3_learner import A3Learner
from scripts.parallel_runners.a3_parallel_runner import A3ParallelRunner
from utils import NameDataset
from torch.utils.data import DataLoader

# 生成pyproject.toml
# pip install poetry
# poetry new my-project
# poetry init

def parse_args(args, parser: argparse.ArgumentParser):
    parser.add_argument("--name_dir", type=str, default=None)
    parser.add_argument("--loss", type=str, default="nll", help="which loss function will be used", choices=["mse","entropy","nll"])
    parser.add_argument("--embed_dim", type=int, default=16, help="rnn hidden dim")
    parser.add_argument("--hidden_size", type=int, default=16, help="rnn hidden dim")
    parser.add_argument("--batch_first", action="store_true", default=False, help="batch first when define rnn using torch.nn")
    parser.add_argument("--bidirectional", action="store_true", default=False, help="use rnn or lstm")
    parser.add_argument("--num_layers", type=int, default=4, help="")
    parser.add_argument("--use_parallel_runner", action="store_false", default=True, help="use different worker to help rnn training")
    parser.add_argument("--num_works", type=int, default=25, help="")
    
    all_args = parser.parse_known_args(args)[0]
    if all_args.use_cuda and torch.cuda.is_available() and all_args.use_parallel_runner == False:
        cuda_rank = all_args.cuda_rank if torch.cuda.device_count() > all_args.cuda_rank else 0
        all_args.device = torch.device(f"cuda:{cuda_rank}")
    else:
        all_args.device = torch.device(f"cpu")
    
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # load names from dataset
    if not os.path.exists(all_args.name_dir):
        logger.error(f"DirNotFoundError: No such directory {all_args.name_dir}")
        sys.exit(-1)
    
    # read male name data
    with open(os.path.join(all_args.name_dir, "male.txt"),"r", encoding="utf-8") as fp:
        male_names = fp.readlines()
        fp.close()
    # read female name data
    with open(os.path.join(all_args.name_dir, "female.txt"),"r", encoding="utf-8") as fp:
        female_names = fp.readlines()
        fp.close()

    # replace "\n" with ""
    male_names = [name.replace("\n","").replace(" ","").lower() for name in male_names]
    female_names = [name.replace("\n","").replace(" ","").lower() for name in female_names]
    names = male_names + female_names
    max_len_name = max([len(name) for name in names])

    char_counter = Counter(''.join(names))
    vocab = sorted(char_counter.keys())
    # add padding & ending flag
    vocab.append("#") # pading flag
    vocab.append("@") # ending flag
    vocab_size = len(vocab)
    
    # add <EOS> && <PAD>
    for i in range(len(names)):
        padding = ["#" for k in range(max_len_name - len(names[i]))]
        padding.append("@")
        for j in range(len(padding)):
            names[i] = names[i] + padding[j]
        if len(names[i]) != max_len_name + 1:
            logger.error(f"padding error!")
            sys.exit(-1)

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    sequences = [[char_to_idx[char] for char in name] for name  in names]

    data_set = NameDataset(sequences = sequences,
                           vocab = vocab,
                           char2idx = char_to_idx,
                           idx2char = idx_to_char,
                        )
    
    train_loader = DataLoader(dataset = data_set, batch_size = all_args.batch_size, shuffle = True)

    if all_args.use_wandb:
        log_name = f'A3_{all_args.seed}_{int(time.time())}'
        wandb.init(
            project = "ml_homework",
            name = log_name,
            group = all_args.wandb_group,
        )
        logger.success(f"use wandb to log data")
    else:
        log_dir = f'./runs/A3_seed{all_args.seed}_{int(time.time())}_{os.getppid()}'
        tb_writer = SummaryWriter(log_dir)
        all_args.tb_writer = tb_writer
        logger.success(f"use tensorboard to log data")   

    
    learner = A3Learner(all_args = all_args, vocab_size = vocab_size, len_name = (max_len_name))

    logger.info(f"training start!")

    if all_args.use_parallel_runner:
        parallel_runer = A3ParallelRunner(all_args, max_len_name, vocab_size)
        parallel_runer.learn(train_loader)
    else:
        learner.learn(train_loader)

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    main(sys.argv[1:])
