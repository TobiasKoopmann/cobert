import numpy as np
import os.path
from functools import partial
from typing import List, Tuple, Optional, Dict
import torch
import copy
import random

from abc import ABC


class Bert4RecDataset(torch.utils.data.dataset.Dataset, ABC):
    def __init__(self,
                 data: List[dict],
                 max_len: Optional[int] = 200,
                 pad_id: int = 0,
                 mask_id: int = 1,
                 ignore_index: int = -100):
        """
        @param data: List of all data points
        @param max_len: Truncation of longer sequences, padding of shorter ones (for batching).
                        Can be 'None' for no truncation.
        @param pad_id: Input ID used for padding, defaults to len(data_ids)
        @param mask_id: Input ID used for masking, defaults to len(data_ids) + 1
        @param ignore_index: (PyTorch's) label ID used for ignoring the loss of this label
        """
        self.data = data
        self.max_len = max_len
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.ignore_index = ignore_index

    def __getitem__(self, i: int) -> Dict:
        """
        Get data point at specific index
        @param i: index of data point
        @return: Dictionary with input_ids, labels, position_ids, attention_mask
        and optionally any additional data
        """
        datapoint = copy.deepcopy(self.data[i])
        a, p, y = self._process_datapoint(datapoint)

        # truncation
        if self.max_len:
            a = a[-self.max_len:]
            p = p[-self.max_len:]
            y = y[-self.max_len:]

        return {
            "author_ids": a,
            "paper_ids": p,
            "labels": y,
            "pad_id": self.pad_id,
            "ignore_index": self.ignore_index,
        }

    def __len__(self) -> int:
        return len(self.data)

    def _process_datapoint(self, datapoint: dict) -> Tuple[List, List, List]:
        """
        Abstract method to handle data specific dataset processing.
        Expected to return (input sequence, labels, OPTIONALLY position ids else return NONE)
        @param datapoint: Single entry of the inherited *self.data*
        @return: (input_sequence, labels, None|position_ids)
        """
        raise NotImplementedError("Abstract base class method called")

    def _get_last_paper_index(self, paper_ids):
        """
        Retrieve the index of the first maximum position id in the sequence (e.g. used for validation/testing)
        @param position_ids: List of position ids
        @return: int index of the first maximum position id
        """
        index = len(paper_ids)
        for paper_id in reversed(paper_ids):
            if paper_id != paper_ids[-1]:
                break
            index -= 1
        else:
            return len(paper_ids) - 1
        return index

    @staticmethod
    def collate(batch):
        """ default collate function"""

        pad_id, ignore_index = batch[0]["pad_id"], batch[0]["ignore_index"]

        # find max length for padding
        max_len = 0
        for bi in batch:
            max_len = max(max_len, len(bi["labels"]))

        author_ids, paper_ids, labels, position_ids, attention_masks = [], [], [], [], []
        for bi in batch:
            # create ids for positional embedding
            positions, last_id, i = [], None, 0
            for paper_id in bi["paper_ids"]:
                if paper_id != last_id:
                    i += 1
                    last_id = paper_id
                positions.append(i)
            position_ids.append(positions + [0] * (max_len - len(positions)))

            # padding of the other embeddings
            author_ids.append(bi["author_ids"] + [pad_id] * (max_len - len(bi["author_ids"])))
            paper_ids.append(bi["paper_ids"] + [pad_id] * (max_len - len(bi["paper_ids"])))
            labels.append(bi["labels"] + [ignore_index] * (max_len - len(bi["labels"])))
            attention_masks.append([1] * len(bi["author_ids"]) + [0] * (max_len - len(bi["author_ids"])))

        return {
            "author_ids": torch.tensor(author_ids, dtype=torch.long),
            "paper_ids": torch.tensor(paper_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
        }

    @staticmethod
    def collate_orig(batch):
        """collate function for no bucket embeddings"""

        pad_id, ignore_index = batch[0]["pad_id"], batch[0]["ignore_index"]

        # find max length for padding
        max_len = 0
        for bi in batch:
            max_len = max(max_len, len(bi["labels"]))

        author_ids, paper_ids, labels, position_ids, attention_masks = [], [], [], [], []
        for bi in batch:
            # create ids for positional embedding
            positions, last_id, i = [], None, 0
            for paper_id in bi["paper_ids"]:
                if paper_id != last_id:
                    i += 1
                    last_id = paper_id
                positions.append(i)
            position_ids.append(list(range(len(positions))) + [0] * (max_len - len(positions)))

            # padding of the other embeddings
            author_ids.append(bi["author_ids"] + [pad_id] * (max_len - len(bi["author_ids"])))
            paper_ids.append(bi["paper_ids"] + [pad_id] * (max_len - len(bi["paper_ids"])))
            labels.append(bi["labels"] + [ignore_index] * (max_len - len(bi["labels"])))
            attention_masks.append([1] * len(bi["author_ids"]) + [0] * (max_len - len(bi["author_ids"])))

        return {
            "author_ids": torch.tensor(author_ids, dtype=torch.long),
            "paper_ids": torch.tensor(paper_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.float),
        }

    @staticmethod
    def collate_seq(batch):
        """ collate function used for sequential author/paper model"""

        pad_id, ignore_index = batch[0]["pad_id"], batch[0]["ignore_index"]

        for bi in batch:
            bi["attention_mask"] = [1] * len(bi["author_ids"])
            positions, last_id, i = [], None, 0
            for paper_id in bi["paper_ids"]:
                if paper_id != last_id:
                    i += 1
                    last_id = paper_id
                positions.append(i)
            bi["position_ids"] = positions

        segment_ids = []
        paper_ids = []
        prev_pos_id = None
        max_len, s_max_len = 0, 0
        for i, bi in enumerate(batch):
            paper_ids.append([0] * len(bi["author_ids"]))
            segment_ids.append([1] * len(bi["author_ids"]))
            for position_id, paper_id in list(zip(bi["position_ids"], bi["paper_ids"])):
                if position_id != prev_pos_id:
                    prev_pos_id = position_id
                    bi["position_ids"].append(position_id)
                    paper_ids[i].append(paper_id)
                    segment_ids[i].append(2)
                    bi["attention_mask"].append(1)
            max_len = max(len(paper_ids[i]) + len(bi["author_ids"]), max_len)
            s_max_len = max(len(bi["author_ids"]), s_max_len)

        author_ids, labels, position_ids, attention_mask = [], [], [], []
        for i, bi in enumerate(batch):
            author_ids.append(bi["author_ids"] + [pad_id] * (s_max_len - len(bi["author_ids"])))
            labels.append(bi["labels"] + [ignore_index] * (max_len - len(bi["labels"])))
            attention_mask.append(bi["attention_mask"] + [0] * (max_len - len(bi["attention_mask"])))
            paper_ids[i] += [pad_id] * (max_len - len(paper_ids[i]))
            segment_ids[i] += [0] * (max_len - len(segment_ids[i]))
            position_ids.append(bi["position_ids"] + [0] * (max_len - len(bi["position_ids"])))

        return {
            "author_ids": torch.tensor(author_ids, dtype=torch.long),
            "paper_ids": torch.tensor(paper_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
        }


class Bert4RecDatasetTrain(Bert4RecDataset):
    def __init__(self,
                 data: List[dict],
                 p_mlm: float = 0.2,
                 p_force_last: float = 0.05,
                 p_mask_max: float = 0.4,
                 **kwargs):
        super().__init__(data, **kwargs)
        self.p_mlm = p_mlm
        self.p_force_last = p_force_last
        self.p_mask_max = p_mask_max

    def _process_datapoint(self, datapoint):
        # index = self._get_last_paper_index(datapoint["paper_ids"]) - 1
        a = datapoint["co_authors"][-self.max_len:]
        y = [self.ignore_index] * len(a)
        a, y = self._mask_sequence(a, y)
        p = datapoint["paper_ids"][-self.max_len:]

        return a, p, y

    def _mask_sequence(self, x, y):
        """
        Masked language modelling. Randomly mask tokens with *probability = self.p_mlm*.
        80% the token is replaced with *self.mask_id*, 10% a random token of the vocab is used, the other 10% are left as they are.
        Specifically for Bert4Rec, with chance *self.p_force_last*, only the LAST token is masked to mimic the test task.
        @param x: List for input ids
        @param y: List for labels
        @return: Both lists, x and y, with randomly masked tokens and respective labels
        """
        if self.p_force_last == 1. or random.random() < self.p_force_last:
            y[-1] = x[-1]
            x[-1] = self.mask_id
        else:
            n_masked = 0
            # iterate randomly to prevent stacking masks at the beginning if p_mask_max is low
            for i in sorted(range(1, len(x)), key=lambda _: random.random()):
                if n_masked < int(self.p_mask_max * len(x)) and random.random() < self.p_mlm:
                    y[i] = x[i]
                    n_masked += 1
                    rnd = random.random()
                    if rnd < 0.1:  # 10% tokens are randomly exchanged
                        x[i] = random.randint(0, len(self.data) - 1)
                    elif rnd < 0.9:  # 80% are masked, the other 10% are left as is
                        x[i] = self.mask_id
            if n_masked == 0:
                i = random.randint(0, len(x) - 1)
                y[i] = x[i]
                x[i] = self.mask_id
        return x, y


class Bert4RecDatasetValidate(Bert4RecDataset):
    def __init__(self, data: List[dict], **kwargs):
        super().__init__(data, **kwargs)

    def _process_datapoint(self, datapoint):
        a = datapoint["co_authors"]
        y = [self.ignore_index] * len(a)
        p = datapoint["paper_ids"]
        for mask_idx in datapoint['masked_ids']:
            y[mask_idx] = a[mask_idx]
            a[mask_idx] = self.mask_id

        return a, p, y


class Bert4RecDatasetTest(Bert4RecDataset):
    def __init__(self, data: List[dict], **kwargs):
        super().__init__(data, **kwargs)

    def _process_datapoint(self, datapoint):
        a = datapoint["co_authors"]
        y = [self.ignore_index] * len(a)
        p = datapoint["paper_ids"]
        for mask_idx in datapoint['masked_ids']:
            if mask_idx > len(a) or mask_idx > len(y):
                print(f"Error: Idx: {mask_idx} - length: {len(a)}/{len(y)} for author {datapoint['author']}")
            y[mask_idx] = a[mask_idx]
            a[mask_idx] = self.mask_id

        return a, p, y
