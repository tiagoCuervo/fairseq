from dataclasses import dataclass
import math
import numpy as np
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    SamePad,
    TransposeLast,
)

N_FREQ_AUDIOGRAM = 8

def pearsonr(x, y):
    # Calculate mean
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # Calculate covariance and variances
    cov = np.mean((x - mean_x) * (y - mean_y))
    var_x = np.var(x)
    var_y = np.var(y)

    # Calculate Pearson correlation coefficient
    pearson_corr = cov / (np.sqrt(var_x) * np.sqrt(var_y))

    return pearson_corr

entropy = lambda x: -sum(x*np.log(x))

def continuousCTCLoss(logits, targets):
    loss = 0
    pred_confusions, resp_confusions = [], []
    for logit, target in zip(logits, targets):
        for freq, word in target:
            ctc = F.ctc_loss(logit, torch.tensor(word), torch.tensor((len(logit),)), torch.tensor((len(word),)))
            loss += freq * ctc
        pred_confusions.append(max([entropy(F.sigmoid(l.cpu().detach())) for l in logit]))
        resp_confusions.append(entropy([freq for freq, word in target]))
    corr = pearsonr(pred_confusions, resp_confusions).statistic
    return loss, corr
        

@dataclass
class Wav2itl_Config(FairseqDataclass):
    # in_dim: int = 1024
    in_dim: int = 1280
    dim: int = 384
    causal: bool = False
    depth: int = 1
    dropout: float = 0.1
    bias: bool = False
    out_dim: int = 42
    nhead: int = 6
    nhead_t: int = 1
    conv_pos_dim: int = 128
    conv_pos_groups: int = 16
    w_cross_attention: bool = True
    huber_loss: bool = False
    ctc_loss: bool = False
    time_hier_transformer: bool = True
    avg_pool: bool = True

@torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ConvPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, d_conv: int, g_conv: int):
        super().__init__()
        self.pe = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv // 2, groups=g_conv)
        std = math.sqrt(4 / (d_conv * d_model))
        nn.init.normal_(self.pe.weight, mean=0, std=std)
        nn.init.constant_(self.pe.bias, 0)
        self.pe = nn.utils.weight_norm(self.pe, name="weight", dim=2)
        self.pe = nn.Sequential(TransposeLast(), self.pe, SamePad(d_conv), nn.GELU(), TransposeLast())

    def forward(self, x):
        return self.pe(x)


class Attention(nn.Module):

    def __init__(self, dim, n_head, bias, dropout, causal, to_self=True):
        super().__init__()
        assert dim % n_head == 0
        self.to_self = to_self
        # key, query, value projections for all heads, but in a batch
        if to_self:
            self.c_attn = nn.Linear(dim, 3 * dim, bias=bias)
        else:
            self.to_q = nn.Linear(dim, dim, bias=bias)
            self.to_k = nn.Linear(dim, dim, bias=bias)
            self.to_v = nn.Linear(dim, dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(dim, dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = dim
        self.dropout = dropout
        self.causal = causal

    def forward(self, x):
        if not self.to_self:
            x, c = x
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.to_self:
            q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        else:
            q, k, v = self.to_q(x), self.to_k(c), self.to_v(c)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # Disable Memory-Efficient Attention kernel because getting backprop error with it, possibly due to gradient penalty loss
        with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    
class MLP(nn.Module):

    def __init__(self, dim, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(dim, 4 * dim, bias=bias)
        self.c_proj  = nn.Linear(4 * dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, dim, n_head, bias, dropout, causal, w_cross_attention):
        super().__init__()
        self.w_cross_attention = w_cross_attention
        self.ln_1 = LayerNorm(dim, bias=bias)
        self.attn = Attention(dim, n_head, bias, dropout, causal)
        if w_cross_attention:
            self.cross_attn = Attention(dim, n_head, bias, dropout, causal, to_self=False)
            self.ln_cross_attn_q = LayerNorm(dim, bias=bias)
            self.ln_cross_attn_kv = LayerNorm(dim, bias=bias)
        self.ln_2 = LayerNorm(dim, bias=bias)
        self.mlp = MLP(dim, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@register_model("wav2itl", dataclass=Wav2itl_Config)
class Wav2itl(BaseFairseqModel):
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1

    def __init__(self, cfg: Wav2itl_Config, target_dict):
        super().__init__()
        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()

        self.cfg = cfg
        inner_dim = cfg.dim
        # Transformer
        self.lin_proj_1 = nn.Linear(cfg.in_dim, inner_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.multi_layer_features = cfg.time_hier_transformer
        self.layer_transformer = TransformerBlock(inner_dim, cfg.nhead, cfg.bias, cfg.dropout, cfg.causal, cfg.w_cross_attention)
        
        self.ln_f_t = LayerNorm(inner_dim, bias=cfg.bias)
        self.ln_f_l = LayerNorm(inner_dim, bias=cfg.bias)

        self.ln_f = LayerNorm(inner_dim, bias=cfg.bias)
        # Merge, pool and predict
        self.pred = nn.Linear(inner_dim, cfg.out_dim, bias=cfg.bias)
        # Aux features input
        self.lis_data_net = nn.Linear(N_FREQ_AUDIOGRAM, inner_dim)
        self.huber_loss, self.ctc_loss = cfg.huber_loss, cfg.ctc_loss

        self.avg_pool = cfg.avg_pool
        if not self.avg_pool:
            self.cls_token = nn.Parameter(torch.randn((inner_dim)))
    
    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.target_dictionary)

    def forward(
        self,
        features,
        padding_mask,
        target,
    ):
        sample_size, max_seq_len, n_layers, inner_dim = features.size()
        
        x = self.dropout(self.lin_proj_1(features))
        padding_mask = torch.repeat_interleave(padding_mask, n_layers, 0)
        down_inner_dim = x.size(-1)
        x = x.view((max_seq_len * sample_size, n_layers, down_inner_dim))
        # x = x.transpose(1, 2).reshape(-1, max_seq_len, down_inner_dim)
        # At this point feats are of size: (batch_size * max_seq_len) x n_layers x down_inner_dim
        l_seq_len = n_layers

        x = torch.cat([self.cls_token.view(1, 1, -1).repeat((sample_size*max_seq_len, 1, 1)), x], dim=1)
        x = self.layer_transformer(x)
        x = self.ln_f_l(x)
        x = x[:, 0]

        x = x.view((sample_size, max_seq_len, down_inner_dim))
        logits = self.pred(x)

        if self.huber_loss:
            preds = F.softmax(logits)
            loss = F.huber_loss(preds * 100, target * 100, reduction="sum")
        elif self.ctc_loss:
            loss, corr = continuousCTCLoss(logits, target)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        result = {"losses": {"itl": loss}, "sample_size": sample_size, "corr":corr}
        return result