import math
import torch
import torch.nn as nn
from linformer import Linformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from nystrom_attention import Nystromformer
from flash_attn.flash_attention import FlashMHA
from mega_pytorch import MegaLayer
from Other_models.S4_model import S4Model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]  # [seq_len, batch_size, dim]
        return self.dropout(x)


class XFormer(nn.Module):
    def __init__(self, model, use_pos, input_size, dim, depth, heads, seq_len):
        super(XFormer, self).__init__()
        self.model = model
        self.use_pos = use_pos
        self.seq_len = seq_len
        self.pos_enc = nn.Embedding(seq_len, dim)
        # self.pos_encoding = PositionalEncoding(dim, seq_len)
        self.linear = nn.Linear(input_size, dim)

        if model == 'transformer':
            encoder_layers = TransformerEncoderLayer(dim, heads, dim)
            self.former = TransformerEncoder(encoder_layers, depth)
        elif model == 'nystromer':
            self.former = Nystromformer(
            dim=dim,
            dim_head=int(dim/heads),
            heads=heads,
            depth=depth,
            num_landmarks=256,  # number of landmarks
            pinv_iterations=6
        )
        elif model == 'linformer':
            self.former = Linformer(
                        dim = dim,
                        seq_len = seq_len,
                        depth =depth,
                        heads = heads,
                        k = dim,
                        one_kv_head = True,
                        share_kv = True
                    )
        elif model == 'flash':
            self.former = FlashMHA(
                embed_dim=dim, # total channels (= num_heads * head_dim)
                num_heads=heads, # number of heads
                dtype=torch.float16,
            )
        elif model == 'mega':
            self.former = nn.Sequential(
                *[MegaLayer(
                    dim = dim,                   # model dimensions
                    ema_heads = heads,              # number of EMA heads
                    attn_dim_qk = dim,            # dimension of queries / keys in attention
                    attn_dim_value = dim*2,        # dimension of values in attention
                    laplacian_attn_fn = False,   # whether to use softmax (false) or laplacian attention activation fn (true)
                ) for _ in range(depth)] 
            )
        elif model == 's4':
            self.former = S4Model(
                            d_input=input_size,
                            d_model=dim,
                            n_layers=depth,
                            dropout=0.2,
                            prenorm=True,
                        )
        print(self.former)

    def forward(self, x):
        # x = x.float()
        positions = torch.arange(0, self.seq_len).expand(x.size(0), self.seq_len).cuda()
        x = self.linear(x)
        # x = x.to(torch.float16)
        if self.use_pos and self.model!="transformer":
            pos_enc =  self.pos_enc(positions)
            x = pos_enc + x

        if self.use_pos and self.model=="transformer":
            x =  self.pos_encoding(x)

        if self.model == 'transformer':
            x = x.permute(1, 0, 2)
            x = self.former(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.former(x)
        return x


class Model4Pretrain(nn.Module):
    def __init__(self, model, depth, heads, input_size, hdim1, hdim2):
        super().__init__()

        self.encoder = XFormer(model, False, input_size, hdim1, depth=depth, heads=heads)
        self.decoder = XFormer(model, False, hdim1, hdim2, depth=depth, heads=heads)
        self.linear = nn.Linear(hdim2, input_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = x.float()
        h = self.encoder(x)
        h = self.decoder(h)
        logits_lm = self.linear(h)
        return self.sig(logits_lm)


class FormerClassifier(nn.Module):
    def __init__(self, name, layers, heads, dim_in, dim_out, clf_dim, output_size, max_len):
        super(FormerClassifier, self).__init__()

        self.encoder = XFormer(name, False, dim_in, dim_out, depth=layers, heads=heads, seq_len=max_len)
        self.Net2 = XFormer(name, False, dim_out, clf_dim, depth=layers, heads=heads, seq_len=max_len)
        self.classifier = nn.Linear(clf_dim, output_size)
        # self.sig = nn.Sigmoid()

    def freeze_cdilNet(self):
        for param in self.cdilNet.parameters():
            param.requires_grad = False

    def forward(self, x1, x2, idx_linear):
        x1, x2 = x1.float(), x2.float()
        idx_linear = idx_linear.to(torch.int64)
        x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
        # x1, x2 = F.pad(x1, (0, 0, 0, 24, 0, 0)), F.pad(x2, (0, 0, 0, 24, 0, 0))
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        y_vcf = y1 - y2
        y_vcf = self.Net2(y_vcf)
        y_class = torch.mean(y_vcf, dim=1)
        y = self.classifier(y_class)
        idx_linear = idx_linear.unsqueeze(0).t()
        y = torch.gather(y, 1, idx_linear)
        return y


class Plant_FormerClassifier(nn.Module):
    def __init__(self, name, layers, heads, dim_in, dim_out, clf_dim, max_seq_len, output_size):
        super().__init__()

        self.encoder = XFormer(name, True, dim_in, dim_out, depth=layers, heads=heads, seq_len=max_seq_len)
        self.Net2 = XFormer(name, False, dim_out, clf_dim, depth=layers, heads=heads, seq_len=max_seq_len)
        self.linear = nn.Linear(clf_dim, output_size).half()

    def forward(self, seq):
        en = self.encoder(seq)
        de = self.Net2(en)
        y_center = de[:, 400:600, :]
        y = self.linear(torch.mean(y_center, dim=1))
        return y
