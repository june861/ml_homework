# -*- encoding: utf-8 -*-
'''
@File    :   rnn.py
@Time    :   2024/11/29 17:52:23
@Author  :   junewluo 
'''

import torch.nn as nn
from utils import MLPLayers

class EmbedLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbedLayer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, token):
        return self.embed(token)


class RNNLayer(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 batch_first,
                 bidirectional = False,
                ):
        super(RNNLayer, self).__init__()
        self.rnn = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first,
            bidirectional = bidirectional,
        )
    
    def forward(self, embeding, last_timesetp = True):
        h_out, hidden_rev = self.rnn(embeding)
        if last_timesetp:
            return h_out[:,-1,:]

        return h_out

class RNNBase(nn.Module):
    def __init__(self,
                 vocab_size, 
                 embed_dim,
                 hidden_size, 
                 num_layers, 
                 batch_first,
                 output_size,
                 activate_func,
                 bidirectional = False,
                ):
        super(RNNBase, self).__init__()

        self.embed_layer = EmbedLayer(vocab_size=  vocab_size, embed_dim =  embed_dim)
        self.rnn_layer = RNNLayer(input_size = embed_dim, 
                                  hidden_size = hidden_size, 
                                  num_layers = num_layers, 
                                  batch_first = batch_first, 
                                  bidirectional = bidirectional
                                )
        self.mlp_layers = MLPLayers(input_dim = hidden_size,
                                    activate_func = activate_func,
                                    hidden_dims = [output_size],
                                )
    

    def forward(self, token, use_log_softmax = True):
        embeding = self.embed_layer(token)
        h_out = self.rnn_layer(embeding)
        fc_out = self.mlp_layers(h_out)

        if use_log_softmax:
            return nn.functional.log_softmax(fc_out, dim=1)
        
        return nn.functional.softmax(fc_out, dim=1)