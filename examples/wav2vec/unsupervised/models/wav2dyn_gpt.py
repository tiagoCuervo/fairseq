# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto
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
from .lm import GPT, GPTConfig, LayerNorm, new_gelu
from omegaconf import MISSING
import os

class SegmentationType(Enum):
    NONE = auto()
    RANDOM = auto()
    UNIFORM_RANDOM = auto()
    UNIFORM_RANDOM_JOIN = auto()
    JOIN = auto()


@dataclass
class SegmentationConfig(FairseqDataclass):
    type: SegmentationType = SegmentationType.NONE
    subsample_rate: float = 0.25
    mean_pool: bool = True
    mean_pool_join: bool = False
    remove_zeros: bool = False


@dataclass
class Wav2vec_UConfig(FairseqDataclass):
    log_gradients: bool = False 
    tunable_gen_lm: bool = False
    tunable_gen_lm_layers: int = -1
    
    discriminator_relativistic: bool = False
    discriminator_type: str = "conv"
    discriminator_dim: int = 256
    discriminator_causal: bool = True
    discriminator_depth: int = 1
    discriminator_max_pool: bool = False
    discriminator_dropout: float = 0.0
    # For transformer type
    discriminator_block_size: int = 1152
    discriminator_bias: bool = False
    discriminator_nhead: int = 6
    # For conv type
    discriminator_kernel: int = 3
    discriminator_dilation: int = 1
    discriminator_linear_emb: bool = False
    discriminator_act_after_linear: bool = False 
    discriminator_spectral_norm: bool = False
    discriminator_weight_norm: bool = False

    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_pad: int = -1
    generator_bias: bool = False
    generator_dropout: float = 0.0
    generator_batch_norm: int = 0
    generator_residual: bool = False

    lm_ckpt: str = MISSING

    blank_weight: float = 0
    blank_mode: str = "add"
    blank_is_sil: bool = False
    no_softmax: bool = False
    gan_on_embeddings: bool = False
    gan_on_posteriograms: bool = True
    compute_genseq_prob: bool = False

    smoothness_weight: float = 0.0
    smoothing: float = 0.0
    smoothing_one_sided: bool = False
    gradient_penalty: float = 0.0
    probabilistic_grad_penalty_slicing: bool = False
    code_penalty: float = 0.0
    mmi_weight: float = 0.0
    target_dim: int = 64
    target_downsample_rate: int = 2
    gumbel: bool = False
    hard_gumbel: bool = True
    temp: Tuple[float, float, float] = (2, 0.1, 0.99995)
    input_dim: int = 128

    segmentation: SegmentationConfig = SegmentationConfig()


class Segmenter(nn.Module):
    cfg: SegmentationConfig

    def __init__(self, cfg: SegmentationConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample_rate = cfg.subsample_rate

    def pre_segment(self, dense_x, dense_padding_mask):
        return dense_x, dense_padding_mask

    def logit_segment(self, logits, padding_mask):
        return logits, padding_mask


class RandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        target_num = math.ceil(dense_x.size(1) * self.subsample_rate)
        ones = torch.ones(dense_x.shape[:-1], device=dense_x.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, dense_x.size(-1))
        dense_x = dense_x.gather(1, indices_ld)
        dense_padding_mask = dense_padding_mask.gather(1, index=indices)
        return dense_x, dense_padding_mask


class UniformRandomSegmenter(Segmenter):
    def pre_segment(self, dense_x, dense_padding_mask):
        bsz, tsz, fsz = dense_x.shape

        target_num = math.ceil(tsz * self.subsample_rate)

        rem = tsz % target_num

        if rem > 0:
            dense_x = F.pad(dense_x, [0, 0, 0, target_num - rem])
            dense_padding_mask = F.pad(
                dense_padding_mask, [0, target_num - rem], value=True
            )

        dense_x = dense_x.view(bsz, target_num, -1, fsz)
        dense_padding_mask = dense_padding_mask.view(bsz, target_num, -1)

        if self.cfg.mean_pool:
            dense_x = dense_x.mean(dim=-2)
            dense_padding_mask = dense_padding_mask.all(dim=-1)
        else:
            ones = torch.ones((bsz, dense_x.size(2)), device=dense_x.device)
            indices = ones.multinomial(1)
            indices = indices.unsqueeze(-1).expand(-1, target_num, -1)
            indices_ld = indices.unsqueeze(-1).expand(-1, -1, -1, fsz)
            dense_x = dense_x.gather(2, indices_ld).reshape(bsz, -1, fsz)
            dense_padding_mask = dense_padding_mask.gather(2, index=indices).reshape(
                bsz, -1
            )
        return dense_x, dense_padding_mask


class JoinSegmenter(Segmenter):
    def logit_segment(self, logits, padding_mask):
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        bsz, tsz, csz = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_tsz = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(bsz, new_tsz, csz)
        new_pad = padding_mask.new_zeros(bsz, new_tsz)

        for b in range(bsz):
            u, idx, c = uniques[b]
            keep = u != -1

            if self.cfg.remove_zeros:
                keep.logical_and_(u != 0)

            if self.training and not self.cfg.mean_pool_join:
                u[0] = 0
                u[1:] = c.cumsum(0)[:-1]
                m = c > 1
                r = torch.rand(m.sum())
                o = (c[m] * r).long()
                u[m] += o
                new_logits[b, : u.numel()] = logits[b, u]
            else:
                new_logits[b].index_add_(
                    dim=0, index=idx.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : c.numel()] /= c.unsqueeze(-1).to(new_logits.device)

            new_sz = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : c.numel()][keep]
                new_logits[b, :new_sz] = kept_logits

            if new_sz < new_tsz:
                pad = new_tsz - new_sz
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True

        return new_logits, new_pad


class UniformRandomJoinSegmenter(UniformRandomSegmenter, JoinSegmenter):
    pass


SEGMENT_FACTORY = {
    SegmentationType.NONE: Segmenter,
    SegmentationType.RANDOM: RandomSegmenter,
    SegmentationType.UNIFORM_RANDOM: UniformRandomSegmenter,
    SegmentationType.UNIFORM_RANDOM_JOIN: UniformRandomJoinSegmenter,
    SegmentationType.JOIN: JoinSegmenter,
}

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x + self.pe[:, :x.size(1)]


class SelfAttention(nn.Module):
    
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
        self.attn = SelfAttention(dim, n_head, bias, dropout, causal)
        self.ln_2 = LayerNorm(dim, bias=bias)
        self.mlp = MLP(dim, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerDiscriminator(nn.Module):
    def __init__(self, in_dim, cfg: Wav2vec_UConfig):
        super().__init__()
        inner_dim = cfg.discriminator_dim
        self.max_pool = cfg.discriminator_max_pool
        self.block_size = cfg.discriminator_block_size
        self.transformer = nn.Sequential(
            nn.Linear(in_dim, inner_dim, bias=False),
            PositionalEncoding(inner_dim, self.block_size),
            nn.Dropout(cfg.discriminator_dropout),
            *[Block(inner_dim, cfg.discriminator_nhead, cfg.discriminator_bias, 
                    cfg.discriminator_dropout, cfg.discriminator_causal) 
                    for _ in range(cfg.discriminator_depth)
            ],
            LayerNorm(inner_dim, bias=cfg.discriminator_bias),
        )
        self.bin_classifier = nn.Linear(inner_dim, 1, bias=cfg.discriminator_bias)

    def forward(self, x, padding_mask):
        t = x.size(1)
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        x = self.transformer(x)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1, keepdim=True)
        if self.max_pool:
            x, _ = x.max(dim=1)
        else:
            x = x.sum(dim=1)
            x = x / x_sz
        return self.bin_classifier(x)


class ConvDiscriminator(nn.Module):
    def __init__(self, in_dim, cfg: Wav2vec_UConfig):
        super().__init__()

        inner_dim = cfg.discriminator_dim
        kernel = cfg.discriminator_kernel
        dilation = cfg.discriminator_dilation
        self.max_pool = cfg.discriminator_max_pool

        if cfg.discriminator_causal:
            padding = kernel - 1
        else:
            padding = kernel // 2

        def make_conv(in_d, out_d, k, p=0, has_dilation=True):
            conv = nn.Conv1d(
                in_d,
                out_d,
                kernel_size=k,
                padding=p,
                dilation=dilation if has_dilation else 1,
            )
            if cfg.discriminator_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            elif cfg.discriminator_weight_norm:
                conv = nn.utils.weight_norm(conv)
            return conv

        inner_net = [
            nn.Sequential(
                make_conv(inner_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
                nn.Dropout(cfg.discriminator_dropout),
                nn.GELU(),
            )
            for _ in range(cfg.discriminator_depth - 1)
        ] + [
            make_conv(inner_dim, 1, kernel, padding, has_dilation=False),
            SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
        ]

        if cfg.discriminator_linear_emb:
            emb_net = [make_conv(in_dim, inner_dim, 1)]
        else:
            emb_net = [
                make_conv(in_dim, inner_dim, kernel, padding),
                SamePad(kernel_size=kernel, causal=cfg.discriminator_causal),
            ]

        if cfg.discriminator_act_after_linear:
            emb_net.append(nn.GELU())

        self.net = nn.Sequential(
            *emb_net,
            nn.Dropout(cfg.discriminator_dropout),
            *inner_net,
        )

    def forward(self, x, padding_mask):
        x = x.transpose(1, 2)  # BTC -> BCT
        x = self.net(x)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)
        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz
        return x


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: Wav2vec_UConfig):
        super().__init__()

        self.cfg = cfg
        self.output_dim = output_dim
        self.stride = cfg.generator_stride
        self.dropout = nn.Dropout(cfg.generator_dropout)
        self.batch_norm = cfg.generator_batch_norm != 0
        self.residual = cfg.generator_residual

        padding = (
            cfg.generator_kernel // 2 if cfg.generator_pad < 0 else cfg.generator_pad
        )
        self.proj = nn.Sequential(
            TransposeLast(),
            nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=cfg.generator_kernel,
                stride=cfg.generator_stride,
                dilation=cfg.generator_dilation,
                padding=padding,
                bias=cfg.generator_bias,
            ),
            TransposeLast(),
        )

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(input_dim)
            self.bn.weight.data.fill_(cfg.generator_batch_norm)
        if self.residual:
            self.in_proj = nn.Linear(input_dim, input_dim)

    def forward(self, dense_x, tokens, dense_padding_mask):
        result = {}

        if self.batch_norm:
            dense_x = self.bn_padded_data(dense_x, dense_padding_mask)
        if self.residual:
            inter_x = self.in_proj(self.dropout(dense_x))
            dense_x = dense_x + inter_x
            result["inter_x"] = inter_x

        dense_x = self.dropout(dense_x)

        dense_x = self.proj(dense_x)
        if self.stride > 1:
            dense_padding_mask = dense_padding_mask[:, :: self.stride]

        if dense_padding_mask.size(1) != dense_x.size(1):
            new_padding = dense_padding_mask.new_zeros(dense_x.shape[:-1])
            diff = new_padding.size(1) - dense_padding_mask.size(1)

            if diff > 0:
                new_padding[:, diff:] = dense_padding_mask
            else:
                assert diff < 0
                new_padding = dense_padding_mask[:, :diff]

            dense_padding_mask = new_padding

        token_x = None
        if tokens is not None:
            token_x = dense_x.new_zeros(tokens.numel(), self.output_dim)
            token_x.scatter_(1, tokens.view(-1, 1).long(), 1)
            token_x = token_x.view(tokens.shape + (self.output_dim,))

        result["dense_x"] = dense_x
        result["token_x"] = token_x
        result["dense_padding_mask"] = dense_padding_mask

        return result

    def bn_padded_data(self, feature, padding_mask):
        normed_feature = feature.clone()
        normed_feature[~padding_mask] = self.bn(
            feature[~padding_mask].unsqueeze(-1)
        ).squeeze(-1)
        return normed_feature


@register_model("wav2dyn_gpt", dataclass=Wav2vec_UConfig)
class Wav2vec_U(BaseFairseqModel):
    def calc_gradient_penalty(self, real_data, fake_data):

        b_size = min(real_data.size(0), fake_data.size(0))
        t_size = min(real_data.size(1), fake_data.size(1))

        if self.cfg.probabilistic_grad_penalty_slicing:

            def get_slice(data, dim, target_size):

                size = data.size(dim)
                diff = size - target_size
                if diff <= 0:
                    return data

                start = np.random.randint(0, diff + 1)
                return data.narrow(dim=dim, start=start, length=target_size)

            real_data = get_slice(real_data, 0, b_size)
            real_data = get_slice(real_data, 1, t_size)
            fake_data = get_slice(fake_data, 0, b_size)
            fake_data = get_slice(fake_data, 1, t_size)

        else:
            real_data = real_data[:b_size, :t_size]
            fake_data = fake_data[:b_size, :t_size]

        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(real_data.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self.discriminator(interpolates, None)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    # def discrim_step(self, num_updates):
    #     return num_updates % 2 == 1
    
    def get_groups_for_update(self, num_updates):
        mod = self.n_groups
        switch = num_updates % mod
        if switch == 0:
            return "generator"
        elif switch == 1:
            return "discriminator"
        elif switch == 2:
            return "lm"

    def __init__(self, cfg: Wav2vec_UConfig, target_dict):
        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()

        super().__init__()

        self.cfg = cfg
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
        self.smoothness_weight = cfg.smoothness_weight

        vocab_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.smoothing = cfg.smoothing
        self.smoothing_one_sided = cfg.smoothing_one_sided
        self.no_softmax = cfg.no_softmax
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel
        self.last_acc = None

        self.gradient_penalty = cfg.gradient_penalty
        self.code_penalty = cfg.code_penalty
        self.mmi_weight = cfg.mmi_weight
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
        assert self.blank_index != target_dict.unk()

        self.pca_A = self.pca_b = None
        d = cfg.input_dim

        self.segmenter = SEGMENT_FACTORY[cfg.segmentation.type](cfg.segmentation)

        # Load GPT model and encoder
        checkpoint = torch.load(cfg.lm_ckpt) # , map_location=p.device)
        if os.path.isfile(os.path.join(os.path.dirname(cfg.lm_ckpt), "dict.txt")):
            print("LM dict provided. Re-ordering LM embeddings to match with data codes")
            with open(os.path.join(os.path.dirname(cfg.lm_ckpt), "dict.txt"), "r") as f:
                lm_vocab = [x.split()[0] for x in f]
            assert set(lm_vocab) == set(target_dict.symbols), "The task and LM vocabularies don't match"
            # Sort list2 using list1 as a key
            sorted_list2 = sorted(lm_vocab, key=lambda x: target_dict.symbols.index(x))
            # Get the indices that would sort list2 to be the same as list1
            indices = [lm_vocab.index(x) for x in sorted_list2]
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.block_size = gptconf.block_size
        state_dict = checkpoint['model']
        # Re-arrange embeddings and output units so that the task codes will match the embeddings codes
        state_dict["transformer.wte.weight"] = state_dict["transformer.wte.weight"][indices]
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"][indices]
        self.ref_lm = GPT(gptconf)
        self.ref_lm.load_state_dict(state_dict)
        self.ref_lm.eval()
        # Freeze reference LM
        for p in self.ref_lm.parameters():
            p.requires_grad = False
        # If required, create separate generator LM and unfreeze tunable layers
        if cfg.tunable_gen_lm:
            self.n_groups = 3
            self.gen_lm = GPT(gptconf)
            self.gen_lm.load_state_dict(state_dict) # Initialize as the reference LM
            if cfg.tunable_gen_lm_layers < 0:  # Tune specific layers 
                # Tune only ln_f (lm_head is not tuned as it shares weights with the token embeddings)
                for name, param in self.gen_lm.named_parameters():
                    if name not in ["transformer.ln_f.weight", "transformer.ln_f.bias"]:
                        param.requires_grad = False
                # Make the specified transformer blocks trainable
                n_blocks_to_update = min(abs(cfg.tunable_gen_lm_layers + 1), len(self.gen_lm.transformer.h))
                if n_blocks_to_update > 0:
                    for i, block in enumerate(self.gen_lm.transformer.h[::-1]):
                        if i < n_blocks_to_update:
                            for param in block.parameters():
                                param.requires_grad = True
            for p in self.gen_lm.parameters():
                p.param_group = "lm"
        else:
            self.n_groups = 2
            self.gen_lm = self.ref_lm

        self.gan_on_embeddings = cfg.gan_on_embeddings
        self.gan_on_posteriograms = cfg.gan_on_posteriograms
        self.compute_genseq_prob = cfg.compute_genseq_prob

        if cfg.discriminator_type == "conv":
            discriminator_model = ConvDiscriminator
        elif cfg.discriminator_type == "transformer":
            discriminator_model = TransformerDiscriminator
        else:
            raise NotImplementedError
        
        if self.gan_on_embeddings:
            disc_in_dim = self.ref_lm.config.n_embd
        elif self.gan_on_posteriograms:
            disc_in_dim = vocab_size
        else:
            disc_in_dim = 1
        self.discriminator = discriminator_model(disc_in_dim, cfg)

        for p in self.discriminator.parameters():
            p.param_group = "discriminator"

        self.discriminator_relativistic = cfg.discriminator_relativistic

        self.generator = Generator(d, vocab_size, cfg)

        for p in self.generator.parameters():
            p.param_group = "generator"

        for p in self.segmenter.parameters():
            p.param_group = "generator"

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0

        if self.mmi_weight > 0:
            self.target_downsample_rate = cfg.target_downsample_rate
            if self.generator.residual:
                self.decoder = nn.Linear(d, cfg.target_dim)
            else:
                self.decoder = nn.Linear(vocab_size, cfg.target_dim)
                self.target_downsample_rate *= 3 # Accounting for downsampling of convolutional generator
            for p in self.decoder.parameters():
                p.param_group = "generator"

        self.log_gradients = cfg.log_gradients

    @classmethod
    def build_model(cls, cfg, task):
        return cls(cfg, task.target_dictionary)

    def get_logits(
        self,
        net_output: Optional[Dict[str, List[Optional[torch.Tensor]]]],
        normalize: bool = False,
    ):
        logits = net_output["logits"]

        if self.blank_weight != 0:
            if self.blank_mode == "add":
                logits[..., self.blank_index] += self.blank_weight
            elif self.blank_mode == "set":
                logits[..., self.blank_index] = self.blank_weight
            else:
                raise Exception(f"invalid blank mode {self.blank_mode}")

        padding = net_output["padding_mask"]
        if padding.any():
            logits[padding] = float("-inf")
            logits[padding][..., self.blank_index] = float("inf")

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits.transpose(0, 1)

    def get_normalized_probs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits = self.get_logits(net_output)

        probs = super().get_normalized_probs(logits, log_probs, sample)
        # BTC -> TBC for ctc
        probs = probs.transpose(0, 1)
        return probs

    def normalize(self, dense_x):

        bsz, tsz, csz = dense_x.shape

        if dense_x.numel() == 0:
            raise Exception(dense_x.shape)
        _, k = dense_x.max(-1)
        hard_x = (
            dense_x.new_zeros(bsz * tsz, csz)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(-1, csz)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        )

        avg_probs = torch.softmax(dense_x.reshape(-1, csz).float(), dim=-1).mean(dim=0)
        prob_perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        )

        if not self.no_softmax:
            if self.training and self.gumbel:
                dense_x = F.gumbel_softmax(
                    dense_x.float(), tau=self.curr_temp, hard=self.hard_gumbel
                ).type_as(dense_x)
            else:
                dense_x = dense_x.softmax(-1)

        return dense_x, code_perplexity, prob_perplexity

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        dense_x_only=False,
        segment=True,
        aux_target=None,
        debugging=False
    ):
        turn = self.get_groups_for_update(self.update_num)
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)

        # if turn == "generator": 
        #     gen_result = self.generator(features, random_label, padding_mask)
        # else:
        #     with torch.no_grad():
        gen_result = self.generator(features, random_label, padding_mask)

        orig_dense_x, token_x = gen_result["dense_x"], gen_result["token_x"]
        orig_dense_padding_mask = gen_result["dense_padding_mask"]

        if segment:
            dense_x, dense_padding_mask = self.segmenter.logit_segment(
                orig_dense_x, orig_dense_padding_mask
            )
        else:
            dense_x = orig_dense_x
            dense_padding_mask = orig_dense_padding_mask

        dense_logits = dense_x
        prob_perplexity = None
        code_perplexity = None

        if not (self.no_softmax and dense_x_only):
            dense_x, code_perplexity, prob_perplexity = self.normalize(dense_logits)

        if dense_x_only or self.discriminator is None:
            return {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }

        token_padding_mask = random_label == self.pad
            
        lm_emb_gen, lm_logits_gen, lm_probs_gen, lm_ent_gen = self.gen_lm(
                dense_x[:, :self.block_size, :],
                self.gan_on_embeddings and not turn == "lm",
                turn == "lm" and self.gan_on_embeddings,
                self.gan_on_posteriograms)
        
        lm_emb_true, lm_logits_true, lm_probs_true, lm_ent_true = self.ref_lm(
                token_x[:, :self.block_size, :],
                self.gan_on_embeddings,
                False,
                self.gan_on_posteriograms)

        disc_t_offset = 0
        if self.gan_on_embeddings:
            disc_in_gen = lm_emb_gen
            disc_in_true = lm_emb_true
        elif self.gan_on_posteriograms:
            if self.compute_genseq_prob:
                disc_t_offset = -1
                disc_in_gen = lm_probs_gen[:, :-1, :] * dense_x[:, 1:self.block_size, :]
                disc_in_true = lm_probs_true[:, :-1, :] * token_x[:, 1:self.block_size, :]
            else:
                disc_in_gen = lm_probs_gen
                disc_in_true = lm_probs_true
        else:
            disc_in_gen = lm_ent_gen
            disc_in_true = lm_ent_true

        dense_y = self.discriminator(disc_in_gen, dense_padding_mask[:, :self.block_size + disc_t_offset])
        token_y = self.discriminator(disc_in_true, token_padding_mask[:, :self.block_size + disc_t_offset])

        
        if self.discriminator_relativistic:
            score_dense = F.sigmoid(dense_y - token_y).mean()
            score_token = F.sigmoid(token_y - dense_y).mean()
        else:
            score_dense = F.sigmoid(dense_y).mean()
            score_token = F.sigmoid(token_y).mean()

        sample_size = features.size(0)

        fake_smooth = self.smoothing
        real_smooth = self.smoothing
        if self.smoothing_one_sided:
            fake_smooth = 0

        smoothness_loss = None
        code_pen = None
        mmi_loss = None
        grad_pen = None
        
        if self.discriminator_relativistic:
            loss_d = None
            loss_g = None
        else:
            loss_token = None
            loss_dense_d = None
            loss_dense_g = None
        
        lm_loss = None
        inter_x = None
        if self.log_gradients:
            grad_dense_lm = None
            grad_dense_g = None
            grad_mmi_g = None
            grad_smoothness_g = None
            grad_code_pen_g = None

        if turn == "discriminator" or debugging:
            if self.discriminator_relativistic:
                # loss_d = F.binary_cross_entropy_with_logits(
                #     (token_y - dense_y),
                #     dense_y.new_ones(dense_y.shape),
                #     reduction="sum",
                # )
                loss_token = F.binary_cross_entropy_with_logits(
                    (token_y - torch.mean(dense_y)),
                    dense_y.new_ones(dense_y.shape),
                    reduction="sum",
                )
                
                loss_dense_d = F.binary_cross_entropy_with_logits(
                    (dense_y - torch.mean(token_y)),
                    dense_y.new_zeros(dense_y.shape),
                    reduction="sum",
                )
                loss_d = (loss_token + loss_dense_d) / 2
            else:
                loss_d = loss_dense_d = F.binary_cross_entropy_with_logits(
                    dense_y,
                    dense_y.new_ones(dense_y.shape) - fake_smooth,
                    reduction="sum",
                )
                loss_token = F.binary_cross_entropy_with_logits(
                    token_y,
                    token_y.new_zeros(token_y.shape) + real_smooth,
                    reduction="sum",
                )
            if self.training and self.gradient_penalty > 0:
                grad_pen = self.calc_gradient_penalty(disc_in_true, disc_in_gen)
                grad_pen = grad_pen.sum() * self.gradient_penalty

            if self.log_gradients:
                grad_dense_lm = torch.norm(autograd.grad(
                    outputs=loss_d,
                    inputs=disc_in_gen,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0])
                grad_dense_g = torch.norm(autograd.grad(
                    outputs=loss_d,
                    inputs=dense_x,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0])
        if turn == "generator" or debugging:
            if self.discriminator_relativistic:
                # loss_g = F.binary_cross_entropy_with_logits(
                #     (dense_y - token_y),
                #     dense_y.new_ones(dense_y.shape),
                #     reduction="sum",
                # )
                loss_dense_g = F.binary_cross_entropy_with_logits(
                    (token_y - torch.mean(dense_y)),
                    dense_y.new_zeros(dense_y.shape),
                    reduction="sum",
                )
                loss_token = F.binary_cross_entropy_with_logits(
                    (dense_y - torch.mean(token_y)),
                    dense_y.new_ones(dense_y.shape),
                    reduction="sum",
                )
                loss_g = (loss_dense_g + loss_token) / 2
            else:
                loss_g = loss_dense_g = F.binary_cross_entropy_with_logits(
                    dense_y,
                    dense_y.new_zeros(dense_y.shape) + fake_smooth,
                    reduction="sum",
                )
            if self.log_gradients:
                grad_dense_lm = torch.norm(autograd.grad(
                    outputs=loss_g,
                    inputs=disc_in_gen,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0])
                grad_dense_g = torch.norm(autograd.grad(
                    outputs=loss_g,
                    inputs=dense_x,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0])
            num_vars = dense_x.size(-1)
            if prob_perplexity is not None:
                code_pen = (num_vars - prob_perplexity) / num_vars
                code_pen = code_pen * sample_size * self.code_penalty
                if self.log_gradients:
                    grad_code_pen_g = torch.norm(autograd.grad(
                        outputs=code_pen,
                        inputs=dense_logits,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0])

            if self.smoothness_weight > 0:
                # smoothness_loss = F.mse_loss(
                    # dense_logits[:, :-1], dense_logits[:, 1:], reduction="none"
                # )
                smoothness_loss = F.mse_loss(
                    orig_dense_x[:, :-1], orig_dense_x[:, 1:], reduction="none"
                )
                # smoothness_loss[dense_padding_mask[:, 1:]] = 0
                smoothness_loss[orig_dense_padding_mask[:, 1:]] = 0
                smoothness_loss = (
                    smoothness_loss.mean() * sample_size * self.smoothness_weight
                )
                if self.log_gradients:
                    grad_smoothness_g = torch.norm(autograd.grad(
                        outputs=smoothness_loss,
                        # inputs=dense_logits,
                        inputs=orig_dense_x,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0])
            if (self.mmi_weight > 0) and (aux_target is not None):
                if self.generator.residual:
                    inter_x = self.decoder(gen_result["inter_x"])
                    inter_mask = padding_mask
                else:
                    inter_x = self.decoder(orig_dense_x)
                    inter_mask = orig_dense_padding_mask
                if self.target_downsample_rate > 1:
                    aux_target = aux_target[:, :: self.target_downsample_rate]  
                max_t_len = min(aux_target.shape[1], inter_x.shape[1])
                mmi_loss = F.cross_entropy(
                    inter_x[:, :max_t_len].transpose(1, 2),
                    aux_target[:, :max_t_len],
                    ignore_index=-1,
                    reduction="none",
                )
                mmi_loss[inter_mask[:, :max_t_len]] = 0
                mmi_loss = mmi_loss.mean() * mmi_loss.shape[0] * self.mmi_weight
                if self.log_gradients:
                    grad_mmi_g = torch.norm(autograd.grad(
                        outputs=mmi_loss,
                        inputs=inter_x,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True
                    )[0])
        if turn == "lm":
            lm_targets = dense_x.argmax(dim=-1)[:, 1:self.block_size].long()
            lm_loss = F.cross_entropy(lm_logits_gen[:, :-1, :].reshape(-1, lm_logits_gen.size(-1)), 
                                      lm_targets.reshape(-1), reduction="none")
            lm_loss_mask = dense_padding_mask[:, :lm_logits_gen.size(1) - 1].reshape(-1)
            lm_loss = lm_loss[~lm_loss_mask]
            lm_loss = lm_loss.mean() # * lm_loss.shape[0] 

        result = {
            "losses": {
                "grad_pen": grad_pen,
                "code_pen": code_pen,
                "smoothness": smoothness_loss,
                "mmi": mmi_loss,
                "lm": lm_loss,
                # "dense_d": loss_dense_d,
                # "dense_g": loss_dense_g,
                # "token_d": loss_token
            },
            "temp": self.curr_temp,
            "code_ppl": code_perplexity,
            "prob_ppl": prob_perplexity,
            "d_steps": int(turn == "discriminator"),
            "sample_size": sample_size,
            "score_dense": score_dense,
            "score_token": score_token
        }

        if debugging:
            result["posteriogram"] = orig_dense_x
            result["posteriogram_mask"] = orig_dense_padding_mask
            result["inter_x"] = inter_x

        if self.discriminator_relativistic:
            result["losses"]["d"] = loss_d
            result["losses"]["g"] = loss_g
        else:
            result["losses"]["dense_d"] = loss_dense_d
            result["losses"]["dense_g"] = loss_dense_g
            result["losses"]["token_d"] = loss_token

        if self.log_gradients:
            result["grad_dense_lm"] = grad_dense_lm
            result["grad_dense_g"] = grad_dense_g
            result["grad_mmi_g"] = grad_mmi_g
            result["grad_smoothness_g"] = grad_smoothness_g
            result["grad_code_pen_g"] = grad_code_pen_g
        return result
