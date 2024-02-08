from dataclasses import dataclass, field
import logging
import math
import os
from typing import Optional, List
import torch
import torch.nn.functional as F

from scipy import stats

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
class ECCCConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory containing embedded audio"}
    )
    max_length: Optional[int] = None
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    target_dictionary: Optional[str] = field(
        default=None,
        metadata={"help": ".txt file with label key to index associations"},
    )
    shuffle: bool = field(default=True, metadata={"help": "shuffle examples"})
    sort_by_length: bool = field(
        default=True, metadata={"help": "sort examples by length of audio timesteps"}
    )

@register_task("eccc", dataclass=ECCCConfig)
class ECCC(FairseqTask):
    """ """

    cfg: ECCCConfig

    def __init__(
        self,
        cfg: ECCCConfig,
        target_dictionary="dict.txt"
    ):
        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()
        
        super().__init__(cfg)
        self._target_dictionary = target_dictionary

    @classmethod
    def setup_task(cls, cfg: ECCCConfig):
        return cls(cfg, target_dictionary=None)

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def valid_step(self, sample, model, criterion):
        res = model(**sample["net_input"])
        # take the max prob of the least confident phoneme in the prediction, and correlate to the max response frequency 
        logging_output = {
            "loss": res['losses']['itl'],
            "corr": res['corr'],
            "sample_size": res['sample_size']
        }

        return res['losses']['itl'], res['sample_size'], logging_output

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        self.datasets[split] = ExtractedFeaturesDataset(
            path=data_path,
            split=split,
            min_length=3,
            max_length=task_cfg.max_length,
            labels=task_cfg.labels,
            label_dict=task_cfg.target_dictionary,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
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
        metrics.log_scalar("corr", corr)

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        model = super().build_model(cfg)

        return model
