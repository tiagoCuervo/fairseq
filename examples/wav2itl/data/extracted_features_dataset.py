# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import contextlib

import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


class ExtractedFeaturesDataset(FairseqDataset):
    def __init__(
        self,
        path,
        split,
        min_length=3,
        max_length=None,
        labels=None,
        shuffle=True,
        sort_by_length=True,
        aux_feats_postfix=None,
    ):
        super().__init__()
        # import ptvsd
        # ptvsd.enable_attach(('0.0.0.0', 7310))
        # print("Attach debugger now")
        # ptvsd.wait_for_attach()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

        self.sizes = []
        self.offsets = []
        self.labels = []
        self.aux_tgt = {}

        path = os.path.join(path, split)
        data_path = path
        self.data = np.load(data_path + ".npy", mmap_mode="r")

        offset = 0
        skipped = 0

        if not os.path.exists(path + f".{labels}"):
            labels = None

        with open(data_path + ".lengths", "r") as len_f, open(
            path + f".{labels}", "r"
        ) if labels is not None else contextlib.ExitStack() as lbl_f:
            for line in len_f:
                length = int(line.rstrip())
                lbl = None if labels is None else next(lbl_f).rstrip().split()
                if length >= min_length and (
                    max_length is None or length <= max_length
                ):
                    self.sizes.append(length)
                    self.offsets.append(offset)
                    if lbl is not None:
                        self.labels.append(list(map(float, lbl)))
                offset += length

        self.sizes = np.asarray(self.sizes)
        self.offsets = np.asarray(self.offsets)

        if aux_feats_postfix is not None:
            for postfix in aux_feats_postfix:
                if not os.path.exists(path+f".{postfix}"):
                    logger.info(f"auxaliry target for {split} missing")
                else:
                    # with open(path+f".{postfix}", "r") as t_f:
                    #     self.aux_tgt[postfix] = [
                    #         torch.Tensor(list(map(float,seg.strip().split())))\
                    #                     for seg in t_f]
                    with open(path + f".{postfix}", "r") as t_f:
                        self.aux_tgt[postfix] = [
                            torch.Tensor(list(map(lambda val: float(val)/100 if postfix == "lis" else float(val), seg.strip().split()))) \
                            for seg in t_f
                        ]
        
        logger.info(f"loaded {len(self.offsets)}, skipped {skipped} samples")

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.data[offset:end].copy()).float()

        res = {"id": index, "features": feats}
        if len(self.labels) > 0:
            res["target"] = self.labels[index]
        
        if self.aux_tgt:
            res["aux_feats"] = {}
            for tgt in self.aux_tgt:
                res["aux_feats"][tgt] = self.aux_tgt[tgt][index]
        return res

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        sizes = [len(s) for s in features]

        target_size = max(sizes)

        if features[0].dim() == 4:
            collated_features = features[0].new_zeros(
                len(features), target_size, 2, features[0].size(-2), features[0].size(-1) # We're using binaural signals from 32 Whisper layers
            )
            padding_mask = torch.BoolTensor(collated_features.shape[:-3]).fill_(False)
        else:
            collated_features = features[0].new_zeros(
                len(features), target_size, 2, features[0].size(-1) # We're using binaural signals
            )
            padding_mask = torch.BoolTensor(collated_features.shape[:-2]).fill_(False)
        for i, (f, size) in enumerate(zip(features, sizes)):
            collated_features[i, :size] = f
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"features": collated_features, "padding_mask": padding_mask},
        }

        if len(self.labels) > 0:
            target = torch.Tensor([s["target"] for s in samples])
            res["net_input"]["target"] = target

        if self.aux_tgt:
            res["net_input"]["aux_feats"] = {}
            for tgt in self.aux_tgt:
                idxs = torch.nn.utils.rnn.pad_sequence(
                    [s["aux_feats"][tgt] for s in samples],
                    batch_first=True,
                    padding_value=-1,
                )
                res["net_input"]["aux_feats"][tgt] = idxs
        
        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]
