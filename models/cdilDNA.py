import math
import json
from typing import NamedTuple
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import BatchNorm1d
import numpy as np


class Config(NamedTuple):
    vocab_size: int = None # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    n_class: int = 919 # Number of classes
    promote: str = "True" #if use special tokens at the beginning
    hdim: int = 128

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CDILBlock(nn.Module):
    def __init__(self, c_in, c_out, hdim, ks, dil, dropout):
        super(CDILBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=hdim, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular', bias = False)
        self.conv2 = nn.Conv1d(in_channels=hdim, out_channels=c_out, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular', bias = False)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hdim)
        self.batch_norm2 = nn.BatchNorm1d(c_out)
        self.res = nn.Conv1d(c_in, c_out, kernel_size=(1,)) if c_in != c_out else None
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.batch_norm1(out)
        out = self.nonlinear(out)
        
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.batch_norm2(out)
        out = self.nonlinear(out)
        res = x if self.res is None else self.res(x)
        return self.nonlinear(out) + res
    
# class CDILBlock2(nn.Module):
#     def __init__(self, c_in, c_out, hdim, ks, dil, dropout):

#         super().__init__()
#         self.conv = nn.Conv1d(in_channels=c_in, out_channels=hdim, kernel_size=ks, padding='same', dilation=dil, padding_mode='circular')

#         self.layer_norm1 = nn.LayerNorm(hdim)
#         self.nonlinear1 = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm2 = nn.LayerNorm(hdim)
#         self.conv21 = nn.Conv1d(in_channels=hdim, out_channels=hdim*2, kernel_size=1)
#         self.nonlinear2 = nn.ReLU()
#         self.conv22 = nn.Conv1d(in_channels=hdim*2, out_channels=c_out, kernel_size=1)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.layer_norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
#         out = self.conv(x)
#         out = self.dropout(self.nonlinear1(out))
#         x2 = out + x
#         x2 = self.layer_norm2(x2.permute(0, 2, 1)).permute(0, 2, 1)
#         out2 = self.dropout2(self.conv22(self.nonlinear2(self.conv21(x2))))
#         return out2 + x2


class CDILLayer(nn.Module):
    def __init__(self, dim_in, dim_out, hdim, ks, dropout):
        super(CDILLayer, self).__init__()
        layers = []
        for i in range(len(dim_out)):
            current_input = dim_in if i == 0 else dim_out[i - 1]
            current_output = dim_out[i]
            hdim = hdim
            current_dilation = 2 ** i
            current_dropout = dropout
            layers += [CDILBlock(current_input, current_output, hdim, ks, current_dilation, current_dropout)]
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_net(x)


class ClassifierHead(nn.Module):
    def __init__(self, dim_hidden, out):
        super(ClassifierHead, self).__init__()
        self.linear = nn.Linear(dim_hidden, out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.normal_(0, 0.01)

    def forward(self, x):
        y = self.linear(x)
        return y


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out, clf_dim, layers, ks, output_size, max_len, dropout):
        super(Classifier, self).__init__()
        self.encoder = CDILLayer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)
        self.revoClf = CDILLayer(dim_out, [clf_dim]*layers, clf_dim*2, ks, dropout)
        self.classifier = ClassifierHead(clf_dim, output_size)
        # self.freeze_cdilNet()dim_in: 5

    def freeze_cdilNet(self):
        for param in self.cdilNet.parameters():
            param.requires_grad = False

    def forward(self, x1, x2, idx_linear):
        # print(x1.shape, x2.shape)
        x1, x2 = x1.float(), x2.float()
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        y = y1 - y2
        y = self.revoClf(y)
        y = torch.mean(y, dim=2)
        y = self.classifier(y)
        idx_linear = idx_linear.unsqueeze(0).t().type(torch.int64)
        y = torch.gather(y, 1, idx_linear)
        return y


class GB_Linear_Classifier(nn.Module):
    """
    The SwanDNA model. Encoder is a stack of SwanDNA blocks. Decoder a global average pooling, followed by a linear layer.

    Args:
        input_size (int): The input size of the embedding layer.
        output_size (int): The output size of the decoder layer.
        max_len (int): The maximum sequence length in the data.
        group_size (int): The size of groups to be shifted.
        hidden_size (int): The hidden layer size for the MLPs.
        mlp_dropout (float): The dropout probability for the MLPs.
        layer_dropout (float): The dropout probability for the SwanDNABlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
    """
    def __init__(self, dim_in, dim_out, layers, ks, output_size, dropout, max_len):
            super().__init__()

            self.encoder = CDILLayer(dim_in, [dim_out]*layers, dim_out*2, ks, dropout)

            self.decoder = nn.Linear(dim_out, output_size)
            # self.freeze_encoder()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.float()
        # x = self.embedding(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.encoder(x)
        x = torch.permute(x, (0, 2, 1))
        x = torch.mean(x, dim=1)
        x = self.decoder(x)
        return x
    
class Model4PretrainCDIL(nn.Module):
    "CDIL Model for Pretrain : Masked LM"
    def __init__(self, dim, hdim1, hdim2, kernel_size, n_layers, dropout):
        super().__init__()
        self.encoder = CDILLayer(dim, [hdim1]*n_layers, hdim1*2, kernel_size, dropout)
        self.hidden_list = [hdim2]*n_layers
        self.hidden_list[-1] = dim
        self.decoder = CDILLayer(hdim1, self.hidden_list, hdim2*2, kernel_size, dropout)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq):
        input_seq = input_seq.float()
        # encoder
        h = torch.permute(input_seq, (0, 2, 1))
        h = self.encoder(h)
        # decoder
        h = self.decoder(h)
        h = torch.permute(h, (0, 2, 1))

        return h