# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum, auto
import math
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from fairseq import utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import TransposeLast
from .lm import GPT, GPTConfig
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
class Wav2uniConfig(FairseqDataclass):
    log_gradients: bool = False 

    lm_ckpt: str = MISSING
    lm_window: int = -1

    generator_kernel: int = 4
    generator_dilation: int = 1
    generator_stride: int = 1
    generator_pad: int = -1
    generator_bias: bool = False
    generator_dropout: float = 0.0
    generator_batch_norm: int = 0
    generator_residual: bool = False

    blank_weight: float = 0
    blank_mode: str = "add"
    blank_is_sil: bool = False
    no_softmax: bool = False

    smoothness_weight: float = 0.0
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


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, cfg: Wav2uniConfig):
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


@register_model("wav2uni", dataclass=Wav2uniConfig)
class Wav2uni(BaseFairseqModel):
    
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.update_num = num_updates
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def __init__(self, cfg: Wav2uniConfig, target_dict):
        super().__init__()

        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()

        self.cfg = cfg
        self.zero_index = target_dict.index("<SIL>") if "<SIL>" in target_dict else 0
        self.smoothness_weight = cfg.smoothness_weight

        output_size = len(target_dict)
        self.pad = target_dict.pad()
        self.eos = target_dict.eos()
        self.no_softmax = cfg.no_softmax
        self.gumbel = cfg.gumbel
        self.hard_gumbel = cfg.hard_gumbel
        self.last_acc = None

        self.code_penalty = cfg.code_penalty
        self.mmi_weight = cfg.mmi_weight
        self.blank_weight = cfg.blank_weight
        self.blank_mode = cfg.blank_mode
        self.blank_index = target_dict.index("<SIL>") if cfg.blank_is_sil else 0
        assert self.blank_index != target_dict.unk()

        self.pca_A = self.pca_b = None
        d = cfg.input_dim

        self.segmenter = SEGMENT_FACTORY[cfg.segmentation.type](cfg.segmentation)

        self.generator = Generator(d, output_size, cfg)

        for p in self.generator.parameters():
            p.param_group = "generator"

        for p in self.segmenter.parameters():
            p.param_group = "generator"

        self.max_temp, self.min_temp, self.temp_decay = cfg.temp
        self.curr_temp = self.max_temp
        self.update_num = 0

        if self.mmi_weight > 0:
            # self.target_downsample_rate = cfg.target_downsample_rate
            # self.decoder = nn.Linear(d, cfg.target_dim)
            # for p in self.decoder.parameters():
            #     p.param_group = "generator"
            self.target_downsample_rate = cfg.target_downsample_rate
            if self.generator.residual:
                self.decoder = nn.Linear(d, cfg.target_dim)
            else:
                self.decoder = nn.Linear(output_size, cfg.target_dim)
                self.target_downsample_rate *= 3 # Accounting for downsampling of convolutional generator
            for p in self.decoder.parameters():
                p.param_group = "generator"

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
        if cfg.lm_window != -1:
            checkpoint["model_args"]["context_width"] = cfg.lm_window
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.block_size = gptconf.block_size
        state_dict = checkpoint['model']
        # Re-arrange embeddings and output units so that the task codes will match the embeddings codes
        state_dict["transformer.wte.weight"] = state_dict["transformer.wte.weight"][indices]
        state_dict["lm_head.weight"] = state_dict["lm_head.weight"][indices]
        self.ref_lm = GPT(gptconf)
        self.ref_lm.load_state_dict(state_dict, strict=True if cfg.lm_window == -1 else False)
        self.ref_lm.eval()
        # Freeze reference LM
        for p in self.ref_lm.parameters():
            p.requires_grad = False

        # with torch.no_grad():
        #     we = self.ref_lm.transformer.wte.weight
        #     wu = self.ref_lm.lm_head.weight
        #     bigram_matrix = we @ wu.T
        #     bigram_matrix[:4, :] -= 1e6
        #     bigram_matrix[:, :4] -= 1e6
        #     bigram_matrix[:4, :4] += 1e6
        #     bigram_matrix = F.softmax(bigram_matrix.reshape(-1), -1).view(len(lm_vocab), len(lm_vocab))
        #     unigram_probs = torch.Tensor([(bigram_matrix[i, :].sum() + bigram_matrix[:, i].sum() - bigram_matrix[i, i]) for i in range(44)]) / 2
        #     self.ref_perplexity = torch.exp(-torch.sum(unigram_probs * torch.log(unigram_probs + 1e-7)))
        #     self.register_buffer("ref_unigram_probs", unigram_probs)
        #     print(f"Reference perplexity: {self.ref_perplexity}")
        
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

        return dense_x, code_perplexity, prob_perplexity, avg_probs

    def forward(
        self,
        features,
        padding_mask,
        random_label=None,
        dense_x_only=False,
        segment=True,
        aux_target=None,
    ):
        if segment:
            features, padding_mask = self.segmenter.pre_segment(features, padding_mask)

        orig_size = features.size(0) * features.size(1) - padding_mask.sum()

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

        dense_logits[..., :4] -= 1e6 # Avoid predicting fairseq standard tokens not present in the corpus

        if not (self.no_softmax and dense_x_only):
            dense_x, code_perplexity, prob_perplexity, unigram_probs = self.normalize(dense_logits)

        if dense_x_only:
            return {
                "logits": dense_x,
                "padding_mask": dense_padding_mask,
            }
        
        ctxt_width = self.ref_lm.config.context_width
        sample_size = dense_x.size(0)
        n_tokens = dense_x.size(1)
        n_windows = n_tokens // ctxt_width

        seqs_lens = (~dense_padding_mask).sum(-1)
        win_starts = torch.randint(low=0, high=n_tokens - ctxt_width, size=(sample_size, n_windows), device=dense_x.device)
        win_starts = (win_starts % seqs_lens.unsqueeze(1))
        start_tokens = torch.gather(dense_logits.argmax(-1), 1, win_starts).view(-1, 1)
        win_idxs = ((win_starts + 1).unsqueeze(2) + 
                    torch.arange(ctxt_width - 1, device=win_starts.device)).view(sample_size, -1)
        win_logits = torch.gather(dense_logits, 1, win_idxs.unsqueeze(-1).expand(-1, -1, dense_logits.size(-1)))
        
        # with torch.no_grad():
        #     lm_emb_gen, lm_logits_gen, lm_probs_gen, lm_ent_gen = self.ref_lm(dense_x[:, :self.block_size, :],
        #                                                                       return_logits=True)
        #     lm_targets = lm_logits_gen[:, :-1, :]
        #     lm_targets[..., :4] -= 1e6 # Avoid predicting fairseq standard tokens not present in the corpus
        #     lm_targets = lm_targets.argmax(-1).view(-1)
        #     # lm_targets = F.softmax(lm_targets, dim=-1).reshape(-1, lm_logits_gen.size(-1))

        # # Minimizing cross-entropy between posteriograms and predictograms
        # lm_loss = F.cross_entropy(dense_logits[:, 1:self.block_size].reshape(-1, dense_logits.size(-1)), 
        #                           lm_targets, reduction="none")
        # lm_loss_mask = dense_padding_mask[:, :dense_logits.size(1) - 1].reshape(-1)
        # lm_loss = lm_loss[~lm_loss_mask]
        # lm_loss = lm_loss.mean() # * lm_loss.shape[0] 

        lm_targets = self.ref_lm.generate(start_tokens, ctxt_width)[:, 1:-1].reshape(-1)
        lm_loss = F.cross_entropy(win_logits.view(-1, win_logits.size(-1)), lm_targets, reduction="mean")


        zero_loss = None
        smoothness_loss = None
        code_pen = None
        mmi_loss = None

        if self.log_gradients:
            grad_lm_g = torch.norm(autograd.grad(
                outputs=lm_loss,
                inputs=dense_logits,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0])
            grad_mmi_g = None
            grad_smoothness_g = None
            grad_code_pen_g = None

        if prob_perplexity is not None:
            num_vars = dense_x.size(-1)
            code_pen = (num_vars - prob_perplexity) / num_vars
            # code_pen = torch.abs(self.ref_perplexity - prob_perplexity) * self.code_penalty
            code_pen = code_pen * sample_size * self.code_penalty
            if self.log_gradients:
                grad_code_pen_g = torch.norm(autograd.grad(
                    outputs=code_pen,
                    inputs=dense_logits,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0])

        # if unigram_probs is not None:
        #     code_pen = -(self.ref_unigram_probs * torch.log(unigram_probs + 1e-12)).sum() * self.code_penalty
        #     if self.log_gradients:
        #         grad_code_pen_g = torch.norm(autograd.grad(
        #             outputs=code_pen,
        #             inputs=dense_logits,
        #             create_graph=True,
        #             retain_graph=True,
        #             only_inputs=True
        #         )[0])

        if self.smoothness_weight != 0:
            smoothness_loss = F.mse_loss(
                dense_logits[:, :-1], dense_logits[:, 1:], reduction="none"
            )
            smoothness_loss[dense_padding_mask[:, 1:]] = 0
            smoothness_loss = (
                smoothness_loss.mean() * sample_size * self.smoothness_weight
            )
            if self.log_gradients:
                grad_smoothness_g = torch.norm(autograd.grad(
                    outputs=smoothness_loss,
                    inputs=dense_logits,
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

        result = {
            "losses": {
                "lm": lm_loss,
                "code_pen": code_pen,
                "smoothness": smoothness_loss,
                "mmi": mmi_loss,
            },
            "temp": self.curr_temp,
            "code_ppl": code_perplexity,
            "prob_ppl": prob_perplexity,
            "sample_size": sample_size
        }

        if self.log_gradients:
            result["grad_lm_g"] = grad_lm_g
            result["grad_mmi_g"] = grad_mmi_g
            result["grad_smoothness_g"] = grad_smoothness_g
            result["grad_code_pen_g"] = grad_code_pen_g
        return result