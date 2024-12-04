# -*- encoding: utf-8 -*-
'''
@File       :a1_parallel_runner.py
@Description:
@Date       :2024/12/03 19:53:07
@Author     :junweiluo
@Version    :python
'''
import wandb
import torch
import pickle
import traceback
import torch.multiprocessing as mp
import torch.distributed as dist
from loguru import logger
from scripts.parallel_runners.base_parallel_runner import ParallelBaseRunner
from utils import RNNBase, get_optimizer, which_loss_criterion

class A3ParallelRunner(ParallelBaseRunner):
    def __init__(self, all_args, len_name, vocab_size):
        super(A3ParallelRunner, self).__init__(all_args)
        
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
        # self.rnn_net = torch.nn.Linear(vocab_size,128)

        self.optimizer = get_optimizer(optim_name = self._optim, net = self.rnn_net, lr = self._lr)
        self.criterion = which_loss_criterion(loss_name = self._loss)
    
    
    def set_share_model(self):
        self.rnn_net.share_memory()
        # self.optimizer.share_memory()
    
    def cal_loss(self, pred_y, true_y):
        if self._loss == "nll":
            true_y = torch.argmax(true_y, dim=1)
        else:
            true_y = true_y.float()
        
        return self.criterion(pred_y, true_y)
    
    
    def update_grad(self, all_gradient):
        self.optimizer.zero_grad()
        
        for params_grads in zip(self.rnn_net.parameters(), zip(*all_gradient)):
            param, grads = params_grads
            avg_grad = torch.stack(grads).mean(dim=0)
            param.grad = avg_grad
        
        self.optimizer.step()
        
    def log_info(self, info):
        logger.info(f"train steps: {info['steps']}  |  train loss: {round(info['train_loss'], 8)}")
        if self._use_wandb:
            wandb.log(info)
        else:
            for k,v in info.items():
                self._tb_writer.add_scalar(k,v)


    # Worker 函数
    def worker(self, rank, model, device, batch_data, pipe_conn):
        # 训练模式
        model.train()
        
        x, y = batch_data
        x = x.to(device)
        y = y.to(device)
        loss = 0.0
            
        for index in range(x.shape[1]):
            x_ = x[:,index,:]
            y_ = y[:,index,:]
            
            net_out = model(x_).float()
            if self._loss == "nll":
                true_y = torch.argmax(y_, dim=1)
            else:
                true_y = y_.float()
            loss += self.criterion(net_out, true_y)
        
        loss /= x.shape[0]
        loss.backward()
        
        gradients = [param.grad.clone().detach() for param in self.rnn_net.parameters()]
        
        # pickle dump
        loss_pkl = pickle.dumps(loss.item())
        gradients_pkl = pickle.dumps(gradients)
        
        pipe_conn.send((loss_pkl, gradients_pkl))
        # pipe_conn.close()

    def wait_process(self, process, parent_conns, all_gradients, total_loss):

        recv_msg = self.recv(parent_conns, self._num_works)
        
        for recv_index, (batch_loss, batch_gradient) in enumerate(recv_msg):
            # pickle load
            batch_loss = pickle.loads(batch_loss)
            batch_gradient = pickle.loads(batch_gradient)
            
            total_loss += batch_loss
            all_gradients.append(batch_gradient)
        
        for p in process:
            p.join()
        
        return total_loss, all_gradients



    def recv(self, parent_conns, up_index):
        for index, parent_conn in enumerate(parent_conns):
            if index < up_index and parent_conn.poll():
                yield parent_conn.recv()
    
    def log_info(self, info):
        logger.info(f"train steps: {info['steps']}  |  train loss: {round(info['train_loss'], 8)}")
        if self._use_wandb:
            wandb.log(info)
        else:
            for k,v in info.items():
                self._tb_writer.add_scalar(k,v)

    def learn(self, train_loader):
        
        self.train_loader = train_loader
        self.set_share_model()
        
        parent_conns, children_conns = zip(*[
            mp.Pipe() for _ in range(self._num_works)
        ])
        
        for epoch in range(self._epochs):
            training_info = {"steps": epoch}
            all_gradients = []
            process = []
            total_loss = 0.0
            conn_index = 0
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                p = mp.Process(target=self.worker, args=(conn_index, self.rnn_net, self._device, batch_data, children_conns[conn_index]))
                # p = mp.Process(target=self.worker, args=(self.rnn_net, self._device, batch_data, children_conns[conn_index]))
                process.append(p)
                p.start()
                conn_index += 1
                
                if len(process) == self._num_works:
                    total_loss, all_gradients = self.wait_process(process, parent_conns, all_gradients, total_loss)
                    process = []
                    conn_index = 0
            
            # dataloader has been fully loaded
            total_loss, all_gradients = self.wait_process(process, parent_conns, all_gradients, total_loss)
            self.update_grad(all_gradients)
            
            training_info["train_loss"] = total_loss / len(self.train_loader)
            
            if epoch % self._log_interval == 0:
                self.log_info(training_info)
            
            logger.info(f"step is {epoch}, loss is {total_loss / len(self.train_loader)}")
        
        for parent_conn in parent_conns:
            parent_conn.close()