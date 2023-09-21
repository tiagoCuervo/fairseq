from dataclasses import dataclass, field
import logging
import math
import os
from typing import Optional, List
import torch
import torch.nn.functional as F

from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from ..data import ExtractedFeaturesDataset

from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed.utils import get_data_parallel_world_size
from omegaconf import MISSING

logger = logging.getLogger(__name__)

@dataclass
class CPC2ReconConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing embedded audio"}
    )
    max_length: Optional[int] = None
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    aux_feats_postfix: List[str] = field(
        default_factory=list,
        metadata={"help": "auxiliary features filename extensions"},
    )
    shuffle: bool = field(default=True, metadata={"help": "shuffle examples"})
    sort_by_length: bool = field(
        default=True, metadata={"help": "sort examples by length of audio timesteps"}
    )

@register_task("cpc2_recon", dataclass=CPC2ReconConfig)
class CPC2Recon(FairseqTask):
    """ """

    cfg: CPC2ReconConfig

    def __init__(
        self,
        cfg: CPC2ReconConfig,
        target_dictionary=None
    ):
        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()
        
        super().__init__(cfg)
        self._target_dictionary = target_dictionary

    @classmethod
    def setup_task(cls, cfg: CPC2ReconConfig):
        return cls(cfg, target_dictionary=None)

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def valid_step(self, sample, model, criterion):
        pred_s, pred_n = model(
            **sample["net_input"],
            preds_only=True,
        )

        targets = sample["net_input"]["target"]
        # target_s, target_n = targets.split(targets.size(-1) // 2, dim=-1)
        target_s, target_n = targets["signal"], targets["noise"]

        # seq_lens = pred_s.size(1) - sample["net_input"]["padding_mask"].sum(1)
        # mses_s = ((pred_s - target_s)**2).mean(2)
        # mses_s[sample["net_input"]["padding_mask"]] = 0            
        # rmse_s = torch.sqrt((mses_s.sum(1) / seq_lens).mean())
        # mses_n = ((pred_n - target_n)**2).mean(2)
        # mses_n[sample["net_input"]["padding_mask"]] = 0            
        # rmse_n = torch.sqrt((mses_n.sum(1) / seq_lens).mean())
        # rmse = 0.5 * (rmse_s + rmse_n)

        preds_cluster_s = pred_s.argmax(-1)
        preds_cluster_n = pred_n.argmax(-1)
        acc_s = ((preds_cluster_s == target_s)[~sample["net_input"]["padding_mask"]]).float().mean()
        acc_n = ((preds_cluster_n == target_n)[~sample["net_input"]["padding_mask"]]).float().mean()
        acc = 0.5 * (acc_s + acc_n)

        sample_size = len(pred_s)

        try:
            world_size = get_data_parallel_world_size()
        except:
            world_size = 1

        logging_output = {
            "loss": acc,
            "acc_n": acc_n,
            "acc_s": acc_s,
            "acc": acc,
            "sample_size": sample_size,
            "_nsamples": sample_size,
            "_world_size": world_size,
            "nsentences": sample_size,
        }

        return acc, sample_size, logging_output

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = ExtractedFeaturesDataset(
            path=data_path,
            split=split,
            min_length=3,
            max_length=task_cfg.max_length,
            labels=task_cfg.labels,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
            aux_feats_postfix=task_cfg.aux_feats_postfix
        )

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        zero = torch.scalar_tensor(0.0)
        # sample_size = sum(log.get("_nsamples", zero) for log in logging_outputs)
        corr = (
            sum(log.get("corr", zero) for log in logging_outputs)
        )
        rmse = (
            sum(log.get("rmse", zero) for log in logging_outputs)
        )
        metrics.log_scalar("corr", corr)
        metrics.log_scalar("rmse", rmse)

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg)

        return model
