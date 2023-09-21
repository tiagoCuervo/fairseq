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
class CPC2Config(FairseqDataclass):
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

@register_task("cpc2", dataclass=CPC2Config)
class CPC2(FairseqTask):
    """ """

    cfg: CPC2Config

    def __init__(
        self,
        cfg: CPC2Config,
        target_dictionary=None
    ):
        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()
        
        super().__init__(cfg)
        self._target_dictionary = target_dictionary

    @classmethod
    def setup_task(cls, cfg: CPC2Config):
        return cls(cfg, target_dictionary=None)

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def valid_step(self, sample, model, criterion):
        preds = model(
            **sample["net_input"],
            preds_only=True,
        )
        targets = sample["net_input"]["target"]
        corr = F.cosine_similarity(preds.view(-1), targets.view(-1), dim=0)
        rmse = torch.sqrt((((preds * 100) - (targets * 100))**2).mean())
        sample_size = len(preds)

        try:
            world_size = get_data_parallel_world_size()
        except:
            world_size = 1

        logging_output = {
            "loss": corr,
            "corr": corr,
            "rmse": rmse,
            "sample_size": sample_size,
            "_nsamples": sample_size,
            "_world_size": world_size,
            "nsentences": sample_size,
        }

        return corr, sample_size, logging_output

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
