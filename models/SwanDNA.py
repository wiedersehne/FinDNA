import math
import torch
import numpy as np
import torch.nn as nn
from flash_pytorch import FLASH, FLASHTransformer
from torch import Tensor

class GEGLU(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """

    def geglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: Tensor) -> Tensor:

        return self.geglu(x)

class Mlp(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module in PyTorch.

    Args:
    in_features (int): Number of input features.
    hidden_features (int, optional): Hidden layer size. Defaults to in_features.
    out_features (int, optional): Output size. Defaults to in_features.
    act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
    drop (float, optional): Dropout probability. Defaults to 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BatchNorm(nn.Module):
    """
    A PyTorch module implementing 1D Batch Normalization for token embeddings.

    Args:
        embedding_size (int): The size of the token embeddings.
    """
    def __init__(self, embedding_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.bn(x)
        x = torch.permute(x, (0, 2, 1))
        return x


class GroupNorm(nn.Module):
    """
    A PyTorch module implementing Group Normalization for token embeddings.

    Args:
        embedding_size (int): The size of the token embeddings.
        n_groups (int): The number of groups to divide the channels into.
    """
    def __init__(self, embedding_size, n_groups):
        super().__init__()
        self.gn = nn.GroupNorm(n_groups, embedding_size)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.gn(x)
        x = torch.permute(x, (0, 2, 1))
        return x


def map_norm(norm_type, embedding_size, group_size=None):
    """
    Maps the given normalization type to the corresponding PyTorch module.

    Args:
        norm_type (str): The normalization type ('LN', 'BN', 'GN', or None).
        embedding_size (int): The size of the token embeddings.
        group_size (int, optional): The number of groups for Group Normalization.

    Returns:
        nn.Module: The corresponding normalization module.
    """
    if norm_type == 'LN':
        norm = nn.LayerNorm(embedding_size)
    elif norm_type == 'BN':
        norm = BatchNorm(embedding_size)
    elif norm_type == 'GN':
        norm = GroupNorm(embedding_size, group_size)
    elif norm_type == 'None':
        norm = nn.Identity()
    return norm


class CircularShift(nn.Module):
    """
    A PyTorch module that performs a parameter-free shift of groups within token embeddings.

    This module can be used to augment or modify the input data in a data-driven manner. The shift is
    performed jointly for all sequences in a batch and is based on powers of 2.

    Args:
        group_size (int): The size of groups to be shifted.
    """
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        y = torch.split(
            tensor=x,
            split_size_or_sections=self.group_size,
            dim=-1
        )

        # Roll sequences in a batch jointly
        # The first group remains unchanged
        z = [y[0]]
        for i in range(1, len(y)):
            offset = - 2 ** (i - 1)
            z.append(torch.roll(y[i], shifts=offset, dims=1))

        z = torch.cat(z, -1)
        return z


class SwanDNABlock(nn.Module):
    """
    A PyTorch module implementing the SwanDNABlock.

    This module combines two main steps in the SwanDNA layer: circular-shift and column_transform.
    The dropout between too is added.

    Args:
        embedding_size (int): The size of the token embeddings.
        group_size (int): The size of groups to be shifted.
        hidden_size (int): The hidden layer size for the MLP.
        mlp_dropout (float): The dropout probability for the MLP.
        layer_dropout (float): The dropout probability for the SwanDNABlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
    """

    def __init__(
        self,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm
    ):
        super().__init__()
        self.prenorm = map_norm(prenorm, embedding_size, group_size)
        self.norm = map_norm(norm, embedding_size, group_size)

        self.column_transform = Mlp(
            embedding_size,
            hidden_size,
            embedding_size,
            act_layer=nn.GELU,
            drop=mlp_dropout
        )

        self.dropout = nn.Dropout(layer_dropout)
        self.shift = CircularShift(group_size)

    def forward(self, x):
        res_con = x
        x = self.prenorm(x)
        x = self.column_transform(x)
        x = self.dropout(x)
        x = self.shift(x)
        x = x + res_con
        # x = self.norm(self.shift(self.dropout(self.column_transform(self.prenorm(x)))) + res_con)
        return x


class SwanDNAEncoder(nn.Module):
    """
    A PyTorch module implementing a SwanDNA Encoder as a stack of SwanDNA layers.
    The number of layers in the stack is determined by the maximum sequence length in the batch.
    The number of layers is fixed for the equal lengths mode.

    Args:
        max_len (int): The maximum sequence length of the input tensor.
        group_size (int): The size of groups to be shifted.
        hidden_size (int): The hidden layer size for the MLP.
        mlp_dropout (float): The dropout probability for the MLP.
        layer_dropout (float): The dropout probability for the SwanDNABlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
    """
    def __init__(
        self,
        max_len,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm
    ):
        super().__init__()
        self.max_len = max_len
        self.max_n_layers = math.ceil(np.log2(max_len))
        self.SwanDNA_blocks = nn.ModuleList(
            [
                SwanDNABlock(
                    embedding_size,
                    group_size,
                    hidden_size,
                    mlp_dropout,
                    layer_dropout,
                    prenorm,
                    norm
                )
                for _ in range(self.max_n_layers)
            ]
        )

    def forward(self, x):
        # If var_len, use a variable number of layers

        for layer in range(self.max_n_layers):
            x = self.SwanDNA_blocks[layer](x)
        return x
    

class SwanDNANetwork(nn.Module):
    """
    A PyTorch module implementing a SwanDNA Encoder as a stack of SwanDNA layers.
    The number of layers in the stack is determined by the maximum sequence length in the batch.
    The number of layers is fixed for the equal lengths mode.

    Args:
        max_len (int): The maximum sequence length of the input tensor.
        group_size (int): The size of groups to be shifted.
        hidden_size (int): The hidden layer size for the MLP.
        mlp_dropout (float): The dropout probability for the MLP.
        layer_dropout (float): The dropout probability for the SwanDNABlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
    """
    def __init__(
        self,
        max_len,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        block_num
    ):
        super().__init__()
        self.blocks = block_num
        self.SwanDNA_blocks = nn.ModuleList(
            [
                SwanDNAEncoder(max_len,
                    embedding_size,
                    group_size,
                    hidden_size,
                    mlp_dropout,
                    layer_dropout,
                    prenorm,
                    norm)
                for _ in range(self.blocks)
            ]
        )

    def forward(self, x):
        # If var_len, use a variable number of layers

        for block in range(self.blocks):
            x = self.SwanDNA_blocks[block](x)
        return x


class Classifier(nn.Module):
    """
    The SwanDNA model. Encoder is a stack of SwanDNA blocks. Decoder a global average pooling, followed by a linear layer.

    Args:
        input_size (int): The input size of the embedding layer.
        output_size (int): The output size of the decoder layer.
        decoder (str): The type of decoder layer. We use 'linear'.
        max_len (int): The maximum sequence length in the data.
        group_size (int): The size of groups to be shifted.
        hidden_size (int): The hidden layer size for the MLPs.
        mlp_dropout (float): The dropout probability for the MLPs.
        layer_dropout (float): The dropout probability for the SwanDNABlock.
        prenorm (str): The type of normalization for the pre-normalization step.
        norm (str): The type of normalization for the post-normalization step.
    """
    def __init__(self,
        input_size,
        output_size,
        max_len,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        coeff
        ):
            super().__init__()
            self.max_n_layers = math.ceil(np.log2(max_len))
            self.embedding = nn.Linear(
                    input_size,
                    embedding_size
            ).apply(self._init_weights)

            self.encoder = SwanDNAEncoder(
                max_len,
                embedding_size,
                group_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            ).apply(self._init_weights)

            self.cm_clf = SwanDNAEncoder(
                max_len,
                embedding_size,
                group_size,
                int(hidden_size*coeff),
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            ).apply(self._init_weights)
            self.decoder = nn.Linear(embedding_size, output_size)

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

    def forward(self, x1, x2, idx_linear):
        x1, x2 = x1.float(), x2.float()
        x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
        x1, x2 = self.embedding(x1), self.embedding(x2)
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        y = y1 - y2
        y = self.cm_clf(y)
        y = torch.mean(y, dim=1)
        y = self.decoder(y)
        idx_linear = idx_linear.unsqueeze(0).t().type(torch.int64)
        y = torch.gather(y, 1, idx_linear)
        return y
    

class GB_Classifier(nn.Module):
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
    def __init__(self,
        input_size,
        output_size,
        max_len,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        coeff
        ):
            super().__init__()
            self.max_n_layers = math.ceil(np.log2(max_len))
            self.group_size = group_size
            self.embedding_size = embedding_size
            self.embedding = nn.Linear(
                    input_size,
                    self.embedding_size
            ).apply(self._init_weights)

            self.encoder = SwanDNAEncoder(
                max_len,
                self.embedding_size,
                group_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            )#.apply(self._init_weights)

            self.cm_clf = SwanDNAEncoder(
                max_len,
                self.embedding_size,
                group_size,
                int(hidden_size*coeff),
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm
            ).apply(self._init_weights)

            self.decoder = nn.Linear(self.embedding_size, output_size)
            self.freeze_encoder()

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
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.cm_clf(x)
        x = torch.mean(x, dim=1)
        x = self.decoder(x)
        return x
    

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
    def __init__(self,
        input_size,
        output_size,
        max_len,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        block_num
        ):
            super().__init__()
            self.max_n_layers = math.ceil(np.log2(max_len))
            self.group_size = group_size
            self.embedding_size = embedding_size
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
                block_num=block_num
            )

            self.decoder = nn.Linear(self.embedding_size, output_size).apply(self._init_weights)

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
        x = self.embedding(x)
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.decoder(x)
        return x
    

class GB_CM_Classifier(nn.Module):
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
    def __init__(self,
        input_size,
        output_size,
        max_len,
        embedding_size,
        group_size,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        prenorm,
        norm,
        coeff
        ):
            super().__init__()
            self.max_n_layers = math.ceil(np.log2(max_len))
            self.group_size = group_size
            self.embedding_size = embedding_size
            self.embedding = nn.Linear(
                    input_size,
                    self.embedding_size
            ).apply(self._init_weights)

            self.encoder = SwanDNANetwork(
                max_len,
                self.embedding_size,
                group_size,
                hidden_size,
                mlp_dropout,
                layer_dropout,
                prenorm,
                norm,
                block_num=5
            )

            self.decoder = nn.Linear(self.embedding_size, output_size)
            self.freeze_encoder()

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
        x = self.embedding(x)
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.decoder(x)
        return x


class GB_Flash_Classifier(nn.Module):
    def __init__(self,
        input_size,
        output_size,
        max_len,
        embedding_size,
        group_size
        ):
            super().__init__()
            self.max_n_layers = 8
            self.group_size = group_size
            self.embedding_size = embedding_size
            self.embedding = nn.Linear(
                    input_size,
                    self.embedding_size
            )
            self.max_len = max_len

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

            self.decoder = nn.Linear(self.embedding_size, output_size)
            # self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.float()
        print(x.shape)
        positions = torch.arange(0, self.max_len).expand(x.size(0), self.max_len).cuda()
        x = self.embedding(x)
        pos_enc =  self.pos_enc(positions)
        x = pos_enc + x
        print(x.shape)
        for layer in range(self.max_n_layers):
            x = self.encoder[layer](x)
        x = torch.mean(x, dim=1)
        x = self.decoder(x)
        return x