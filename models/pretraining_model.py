from models.SwanDNA import SwanDNANetwork
import torch.nn as nn
import math
from flash_pytorch import FLASH, FLASHTransformer
import numpy as np
import torch

class Model4Pretrain(nn.Module):
    """
    SwanDNA Model for Pretrain : Masked LM
    With one SwanDNA encoder and one SwanDNA decoder.
    """
    def __init__(self, input_size, max_len, embedding_size, group_size, hidden_size, mlp_dropout, layer_dropout, prenorm, norm):
        super().__init__()
        self.max_n_layers = math.ceil(np.log2(max_len))
        self.embedding_size = (self.max_n_layers+1) * group_size
        self.embedding = nn.Linear(
                    input_size,
                    self.embedding_size
            )
        self.encoder = SwanDNANetwork(
                max_len,
                self.embedding_size,
                group_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm,
                4
            )
        # self.decoder = SwanDNAEncoder(
        #         max_len,
        #         self.embedding_size,
        #         group_size,
        #         hidden_size,
        #         mlp_dropout,
        #         layer_dropout,
        #         prenorm,
        #         norm
        #     )

        self.linear = nn.Linear(self.embedding_size, input_size)

    def forward(self, input_seq):
        input_seq = input_seq.float()
        h = self.embedding(input_seq)
        # encoder
        h = self.encoder(h)
        # decoder
        h = self.linear(h)

        return h
    

class Model4TSNE(nn.Module):
    """
    SwanDNA Model for Pretrain : Masked LM
    With one SwanDNA encoder and one SwanDNA decoder.
    """
    def __init__(self, input_size, max_len, embedding_size, track_size, hidden_size, mlp_dropout, layer_dropout, prenorm, norm):
        super().__init__()
        self.max_n_layers = math.ceil(np.log2(max_len))
        self.embedding_size = (self.max_n_layers+1) * track_size
        self.embedding = nn.Linear(
                    input_size,
                    self.embedding_size
            )
        self.encoder = SwanDNANetwork(
                max_len,
                self.embedding_size,
                track_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            )

    def forward(self, input_seq):
        input_seq = input_seq.float()
        h = self.embedding(input_seq)
        # encoder
        h = self.encoder(h)

        return h
    

class Model4PretrainFlash(nn.Module):
    """
    SwanDNA Model for Pretrain : Masked LM
    With one SwanDNA encoder and one SwanDNA decoder.
    """
    def __init__(self, input_size, embedding_size, group_size, max_len):
        super().__init__()
        self.max_n_layers = 8
        self.max_len = max_len
        self.embedding = nn.Linear(
                    input_size,
                    embedding_size
            )
        
        self.pos_enc = nn.Embedding(max_len, embedding_size)

        self.encoder = nn.ModuleList(
            [
                FLASH(
                    dim = embedding_size,
                    group_size = group_size,             # group size
                    causal = True,                # autoregressive or not
                    query_key_dim = int(embedding_size/4),          # query / key dimension
                    expansion_factor = 2.,        # hidden dimension = dim * expansion_factor
                    laplace_attn_fn = True   # new Mega paper claims this is more stable than relu squared as attention function
                )
                for _ in range(self.max_n_layers)
            ]
        )

        self.linear = nn.Linear(embedding_size, input_size)

    def forward(self, input_seq):
        input_seq = input_seq.float()
        positions = torch.arange(0, self.max_len).expand(input_seq.size(0), self.max_len).cuda()
        h = self.embedding(input_seq)
        pos_enc =  self.pos_enc(positions)
        h = pos_enc + h
        # input_seq = torch.permute(input_seq, (0, 2, 1))
        for layer in range(self.max_n_layers):
            h = self.encoder[layer](h)
        # h = torch.permute(h, (0, 2, 1))
        # decoder
        h = self.linear(h)

        return h