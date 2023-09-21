from dataclasses import dataclass
import math
import numpy as np
from typing import Tuple, List, Optional, Dict
from omegaconf import MISSING

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

N_MFCC_FILTERS = 39
N_CLUSTERS = 500

@dataclass
class Wav2sn_Config(FairseqDataclass):
    in_dim: int = 1280
    dim: int = 384
    causal: bool = False
    dropout: float = 0.1
    bias: bool = False
    nhead: int = 6
    huber_loss: bool = True
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

    def __init__(self, dim, n_head, bias, dropout, causal):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(dim, 3 * dim, bias=bias)
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
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
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

class Block(nn.Module):

    def __init__(self, dim, n_head, bias, dropout, causal):
        super().__init__()
        self.ln_1 = LayerNorm(dim, bias=bias)
        self.attn = Attention(dim, n_head, bias, dropout, causal)
        self.ln_2 = LayerNorm(dim, bias=bias)
        self.mlp = MLP(dim, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@register_model("wav2sn", dataclass=Wav2sn_Config)
class Wav2sn(BaseFairseqModel):
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates

    def discrim_step(self, num_updates):
        return num_updates % 2 == 1

    def __init__(self, cfg: Wav2sn_Config, target_dict):
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
        self.layer_transformer = Block(inner_dim, cfg.nhead, cfg.bias, cfg.dropout, cfg.causal)
        self.ln_f = LayerNorm(inner_dim, bias=cfg.bias)
        
        self.huber_loss = cfg.huber_loss

        # Merge, pool and predict
        self.pred_signal = nn.Linear(inner_dim, N_MFCC_FILTERS if self.huber_loss else N_CLUSTERS, bias=cfg.bias)
        self.pred_noise = nn.Linear(inner_dim, N_MFCC_FILTERS if self.huber_loss else N_CLUSTERS, bias=cfg.bias)
            
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
        target=None,
        preds_only=False,
        aux_feats=None,
    ):
        sample_size, max_seq_len, n_layers, inner_dim = features.size()
        x = self.dropout(self.lin_proj_1(features))
        down_inner_dim = x.size(-1)
        x = x.view(sample_size * max_seq_len, n_layers, down_inner_dim)
        x = torch.cat([self.cls_token.view(1, 1, -1).repeat((sample_size * max_seq_len, 1, 1)), x], dim=1)
        x = self.layer_transformer(x)
        x = self.ln_f(x)
        x = x[:, 0]
        x = x.view(sample_size, max_seq_len, down_inner_dim)
    
        pred_s = self.pred_signal(x)
        pred_n = self.pred_noise(x)
        
        if preds_only:
            return pred_s, pred_n
        
        # target_s, target_n = target.split(target.size(-1) // 2, dim=-1)
        target_s, target_n = target["signal"], target["noise"]
        
        if self.huber_loss:
            losses = (F.huber_loss(pred_s, target_s, reduction="none") + 
                    F.huber_loss(pred_n, target_n, reduction="none")).mean(2)
            losses[padding_mask] = 0
            loss = 0.5 * losses.sum()
            
            with torch.no_grad():
                seq_lens = max_seq_len - padding_mask.sum(1)
                mses_s = ((pred_s - target_s)**2).mean(2)
                mses_s[padding_mask] = 0            
                rmse_s = torch.sqrt((mses_s.sum(1) / seq_lens).mean())
                mses_n = ((pred_n - target_n)**2).mean(2)
                mses_n[padding_mask] = 0            
                rmse_n = torch.sqrt((mses_n.sum(1) / seq_lens).mean())
            result = {
                "losses": {
                    "itl": loss
                },
                "sample_size": sample_size,
                "rmse_s": rmse_s,
                "rmse_n": rmse_n,
                "rmse": 0.5 * (rmse_s + rmse_n)
            }
        else:
            pred_s = pred_s.view(-1, N_CLUSTERS)
            pred_n = pred_n.view(-1, N_CLUSTERS)    
            target_s = target_s.view(-1)
            target_n = target_n.view(-1)
            loss = F.cross_entropy(pred_s, target_s, reduction="none") + F.cross_entropy(pred_n, target_n, reduction="none")
            loss[padding_mask.view(-1)] = 0
            loss = 0.5 * loss.sum()
            with torch.no_grad():
                preds_cluster_s = pred_s.argmax(-1)
                preds_cluster_n = pred_n.argmax(-1)
                acc_s = ((preds_cluster_s == target_s)[~padding_mask.view(-1)]).float().mean()
                acc_n = ((preds_cluster_n == target_n)[~padding_mask.view(-1)]).float().mean()
            result = {
                "losses": {
                    "itl": loss
                },
                "sample_size": sample_size,
                "acc_s": acc_s,
                "acc_n": acc_n,
                "acc": 0.5 * (acc_s + acc_n)
            }
        return result