# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os, nltk
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
        label_dict=None,
        shuffle=True,
        sort_by_length=True,
        aux_target_postfix=None,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.max_label_length = 12
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.label_dict = {l[0]:int(l[1]) for l in np.loadtxt(os.path.join(path, label_dict), delimiter=' ', dtype=str)}
        if labels is not None:
            assert label_dict is not None

        self.sizes = []
        self.offsets = []
        self.labels = []
        self.aux_tgt = None

        path = os.path.join(path, split)
        data_path = path
        self.data = np.load(data_path + ".npy", mmap_mode="r")

        skipped = 0
        offset = 0
        self.to_phoneme = nltk.corpus.cmudict.dict()

        if not os.path.exists(path + f".{labels}"):
            labels = None

        with open(data_path + ".lengths", "r") as len_f, open(
            path + f".{labels}", "r"
        ) if labels is not None else contextlib.ExitStack() as lbl_f:
            next(lbl_f)
            for line in len_f:
                length = int(line.rstrip())
                lbl = None if labels is None else next(lbl_f).rstrip().split(',')
                responses, counts, tot = lbl[9].split('|'), lbl[11].split(' '), int(lbl[10])
                if length >= min_length and (
                    max_length is None or length <= max_length
                ) and np.all([r in self.to_phoneme for r in responses]):
                    self.labels.append((responses, counts, tot))
                    self.sizes.append(length)
                    self.offsets.append(offset)
                    offset += length
                else:
                    skipped += 1
                
        self.sizes = np.asarray(self.sizes)
        
        if aux_target_postfix is not None:
            if not os.path.exists(path+f".{aux_target_postfix}"):
                logger.info(f"auxaliry target for {split} missing")
            else:
                with open(path+f".{aux_target_postfix}", "r") as t_f:
                    self.aux_tgt = [
                        torch.LongTensor(list(map(int,seg.strip().split())))\
                                    for seg in t_f]
 
        logger.info(f"loaded {len(self.offsets)}, skipped {skipped} samples")

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.data[offset:end].copy()).float()

        res = {"id": index, "features": feats}
        if len(self.labels) > 0:
            responses, counts, tot = self.labels[index]
            phones = torch.zeros((len(responses), self.max_label_length), dtype=int)
            for i, response in enumerate(responses):
                tr_response = [self.label_dict[phone] for phone in self.to_phoneme[response][0]]
                phones[i, :len(tr_response)] = torch.tensor(tr_response)
            res["target"] = {'responses':phones, 'freqs':np.array(counts).astype(int)/tot, 'lengths':(phones!=0).sum(axis=-1)}

        return res

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        sizes = [len(s) for s in features]
        target_size = max(sizes)

        collated_features = features[0].new_zeros(
            len(features), target_size, *features[0].shape[-2:]
        )
        padding_mask = torch.BoolTensor(collated_features.shape[:2]).fill_(False)
        for i, (f, size) in enumerate(zip(features, sizes)):
            collated_features[i, :size] = f
            padding_mask[i, size:] = True

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {"features": collated_features, "padding_mask": padding_mask},
        }

        if len(self.labels) > 0:
            res['net_input']["target"] = {
                'responses': torch.vstack([s['target']['responses'] for s in samples]),
                'freqs': torch.concat([torch.tensor(s['target']['freqs']) for s in samples]),
                'lengths': torch.concat([s['target']['lengths'] for s in samples]),
                'n_responses': torch.tensor([len(s['target']['responses']) for s in samples], dtype=int)
            }
        
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
