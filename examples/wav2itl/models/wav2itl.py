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
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # Calculate covariance and variances
    cov = torch.mean((x - mean_x) * (y - mean_y))
    var_x = torch.var(x)
    var_y = torch.var(y)

    # Calculate Pearson correlation coefficient
    pearson_corr = cov / (torch.sqrt(var_x) * torch.sqrt(var_y))

    return pearson_corr

@dataclass
class Wav2itl_Config(FairseqDataclass):
    # in_dim: int = 1024
    in_dim: int = 1280
    dim: int = 384
    causal: bool = False
    depth: int = 1
    dropout: float = 0.1
    bias: bool = False
    nhead: int = 6
    nhead_t: int = 1
    conv_pos_dim: int = 128
    conv_pos_groups: int = 16
    w_cross_attention: bool = True
    huber_loss: bool = True
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

class BinauralBlock(nn.Module):

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
        x_l, x_r = x
        x_l = x_l + self.attn(self.ln_1(x_l))
        x_r = x_r + self.attn(self.ln_1(x_r))
        if self.w_cross_attention:
            x_l = x_l + self.cross_attn((self.ln_cross_attn_q(x_l), self.ln_cross_attn_kv(x_r)))
            x_r = x_r + self.cross_attn((self.ln_cross_attn_q(x_r), self.ln_cross_attn_kv(x_l)))
        x_l = x_l + self.mlp(self.ln_2(x_l))
        x_r = x_r + self.mlp(self.ln_2(x_r))
        return (x_l, x_r)


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
        if cfg.time_hier_transformer:
            self.temporal_transformer = BinauralBlock(inner_dim, cfg.nhead_t, cfg.bias, cfg.dropout, cfg.causal, cfg.w_cross_attention)
            self.layer_transformer = BinauralBlock(inner_dim, cfg.nhead, cfg.bias, cfg.dropout, cfg.causal, cfg.w_cross_attention)
            # self.temporal_transformer = nn.Sequential(*[BinauralBlock(inner_dim, cfg.nhead_t, cfg.bias, 
            #                                                 cfg.dropout, cfg.causal, cfg.w_cross_attention) 
            #                                                 for _ in range(cfg.depth)])
            # self.layer_transformer = nn.Sequential(*[BinauralBlock(inner_dim, cfg.nhead, cfg.bias, 
            #                                                 cfg.dropout, cfg.causal, cfg.w_cross_attention) 
            #                                                 for _ in range(cfg.depth)])
            self.ln_f_t = LayerNorm(inner_dim, bias=cfg.bias)
            self.ln_f_l = LayerNorm(inner_dim, bias=cfg.bias)
        else:
            self.pos_enc = ConvPositionalEncoding(inner_dim, cfg.conv_pos_dim, cfg.conv_pos_groups)
            self.transformer_blocks = nn.Sequential(*[BinauralBlock(inner_dim, cfg.nhead, cfg.bias, 
                                                            cfg.dropout, cfg.causal, cfg.w_cross_attention) 
                                                            for _ in range(cfg.depth)])
        self.ln_f = LayerNorm(inner_dim, bias=cfg.bias)
        # Merge, pool and predict
        self.merge_layer = nn.Linear(inner_dim * 2, inner_dim)
        self.pred = nn.Linear(inner_dim, 1, bias=cfg.bias)
        # Aux features input
        self.lis_data_net = nn.Linear(N_FREQ_AUDIOGRAM, inner_dim)
        self.huber_loss = cfg.huber_loss

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
        if self.multi_layer_features:
            sample_size, max_seq_len, n_audio_channels, n_layers, inner_dim = features.size()
        else:
            sample_size, max_seq_len, n_audio_channels, inner_dim = features.size()
        l_features, r_features = features.split(1, dim=2)
        if self.multi_layer_features:
            x_l = self.dropout(self.lin_proj_1(l_features.squeeze(2)))
            x_r = self.dropout(self.lin_proj_1(r_features.squeeze(2)))
            padding_mask = torch.repeat_interleave(padding_mask, n_layers, 0)
            down_inner_dim = x_l.size(-1)
            # At this point feats are of size: sample_size x max_seq_len x n_layers x down_inner_dim
            x_l = x_l.transpose(1, 2).reshape(-1, max_seq_len, down_inner_dim)
            x_r = x_r.transpose(1, 2).reshape(-1, max_seq_len, down_inner_dim)
            x_l, x_r = self.temporal_transformer((x_l, x_r))
            x_l = self.ln_f_t(x_l)
            x_r = self.ln_f_t(x_r)
            # At this point feats are of size: (sample_size * n_layers) x max_seq_len x down_inner_dim
            # We do average pooling across time
            x_l[padding_mask] = 0
            x_r[padding_mask] = 0
            x_sz = features.size(1) - padding_mask.sum(dim=-1, keepdim=True)
            x_l = x_l.sum(dim=1) / x_sz
            x_r = x_r.sum(dim=1) / x_sz
            # At this point feats are of size: (sample_size * n_layers) x down_inner_dim
            x_l = x_l.view(sample_size, n_layers, down_inner_dim)
            x_r = x_r.view(sample_size, n_layers, down_inner_dim)
            l_seq_len = n_layers
            if aux_feats is not None:
                if "lis" in aux_feats:
                    l_lis_features, r_lis_features = aux_feats["lis"].split(aux_feats["lis"].size(-1) // 2, dim=1)
                    lis_l = self.lis_data_net(l_lis_features)
                    lis_r = self.lis_data_net(r_lis_features)
                    x_l = torch.cat([lis_l.unsqueeze(1), x_l], dim=1)
                    x_r = torch.cat([lis_r.unsqueeze(1), x_r], dim=1)
                    l_seq_len += 1
            if self.avg_pool:
                x_l, x_r = self.layer_transformer((x_l, x_r))
                x_l = self.ln_f_l(x_l)
                x_r = self.ln_f_l(x_r)
                x_l = x_l.sum(dim=1) / l_seq_len
                x_r = x_r.sum(dim=1) / l_seq_len
            else:
                x_l = torch.cat([self.cls_token.view(1, 1, -1).repeat((sample_size, 1, 1)), x_l], dim=1)
                x_r = torch.cat([self.cls_token.view(1, 1, -1).repeat((sample_size, 1, 1)), x_r], dim=1)
                x_l, x_r = self.layer_transformer((x_l, x_r))
                x_l = self.ln_f_l(x_l)
                x_r = self.ln_f_l(x_r)
                x_l = x_l[:, 0]
                x_r = x_r[:, 0]
        else:
            x_l = self.dropout(self.pos_enc(self.lin_proj_1(l_features.squeeze(2))))
            x_r = self.dropout(self.pos_enc(self.lin_proj_1(r_features.squeeze(2)))) 
            seq_offset = 0
            if aux_feats is not None:
                if "lis" in aux_feats:
                    l_lis_features, r_lis_features = aux_feats["lis"].split(aux_feats["lis"].size(-1) // 2, dim=1)
                    lis_l = self.lis_data_net(l_lis_features)
                    lis_r = self.lis_data_net(r_lis_features)
                    x_l = torch.cat([lis_l.unsqueeze(1), x_l], dim=1)
                    x_r = torch.cat([lis_r.unsqueeze(1), x_r], dim=1)
                    seq_offset = 1
            x_l, x_r = self.transformer_blocks((x_l, x_r))
            x_l = self.ln_f(x_l[:, seq_offset:])
            x_r = self.ln_f(x_r[:, seq_offset:])
            x_l[padding_mask] = 0
            x_r[padding_mask] = 0
            x_sz = features.size(1) - padding_mask.sum(dim=-1, keepdim=True)
            x_l = x_l.sum(dim=1) / x_sz
            x_r = x_r.sum(dim=1) / x_sz
        x = torch.cat([x_l, x_r], dim=-1)
        x = self.merge_layer(x)
        logits = self.pred(x)
        preds = F.sigmoid(logits)

        if preds_only:
            return preds
        if self.huber_loss:
            loss = F.huber_loss(preds * 100, target * 100, reduction="sum")
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        with torch.no_grad():
            corr = pearsonr(preds.view(-1), target.view(-1))
            rmse = torch.sqrt((((preds * 100) - (target * 100))**2).mean())
        result = {
            "losses": {
                "itl": loss
            },
            "sample_size": sample_size,
            "corr": corr,
            "rmse": rmse
        }
        return result