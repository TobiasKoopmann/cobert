import os
import json

import torch
import logging
from typing import Iterable

from torch.utils.data import DataLoader

from util import Bert4RecTask
from util.callbacks import Evaluation, PredictionSerializer

# from model import Bert4RecOG, Bert4RecReformed
from model.bert import *
from model.dataset import *


def get_model(args: dict):
    config = args["data_dir"], args["hidden_size"], args["n_layers"], args["n_heads"], args["max_len"], args["dropout"]
    if args["og-model"]:
        logging.info("load og model")
        model = get_og_model(*config)
        assert not args["paper_embedding"] and not args["pretrained_paper_embedding"], "OG model cannot use paper embedding"
    elif args["nova-model"]:
        logging.info("load nova model")
        model = get_nova_model(*config)
    elif args["seq-model"]:
        logging.info("load sequential model")
        model = get_seq_model(*config)
    elif not (args["paper_embedding"] or args["pretrained_paper_embedding"]) and not args["weighted_embedding"]:
        logging.info("load ae")
        model = get_ae_model(*config)
    elif not (args["paper_embedding"] or args["pretrained_paper_embedding"]) and args["weighted_embedding"]:
        logging.info("load awe model")
        model = get_aew_model(*config)
    elif (args["paper_embedding"] or args["pretrained_paper_embedding"]) and not args["weighted_embedding"]:
        logging.info("load cobert")
        model = get_aepe_model(*config)
    elif (args["paper_embedding"] or args["pretrained_paper_embedding"]) and args["weighted_embedding"]:
        logging.info("load ae pe weighted")
        model = get_aepew_model(*config)
    else:
        raise ValueError("Invalid configuration")

    if args["pretrained_author_embedding"]:
        logging.info("load pretrained author embedding")
        authors_emb_file = _get_authors_emb_file(args["data_dir"])
        model.embedding.token.load_state_dict(authors_emb_file)

    if args["pretrained_paper_embedding"]:
        logging.info("load pretrained paper embedding")
        papers_emb_file = _get_papers_emb_file(args["data_dir"])
        model.embedding.paper.load_state_dict(papers_emb_file)

    return model


def get_og_model(data_dir: str,
                 hidden_size: int = 768,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 max_len: int = 200,
                 dropout: float = 0.1):
    authors = _get_authors_file(data_dir)

    model = Bert4RecOG(vocab_size=len(authors) + 2,  # +2 for padding and mask
                       hidden_size=hidden_size,
                       n_layers=n_layers,
                       n_heads=n_heads,
                       max_len=max_len + 1,  # +1 for padding
                       p_dropout=dropout)

    return model


def get_ae_model(data_dir: str,
                 hidden_size: int = 768,
                 n_layers: int = 2,
                 n_heads: int = 8,
                 max_len: int = 200,
                 dropout: float = 0.1):
    authors = _get_authors_file(data_dir)

    model = Bert4RecAE(vocab_size=len(authors) + 2,  # +2 for padding and mask
                       hidden_size=hidden_size,
                       n_layers=n_layers,
                       n_heads=n_heads,
                       max_len=max_len + 1,  # +1 for padding
                       p_dropout=dropout)

    return model


def get_aew_model(data_dir: str,
                  hidden_size: int = 768,
                  n_layers: int = 2,
                  n_heads: int = 8,
                  max_len: int = 200,
                  dropout: float = 0.1):
    authors = _get_authors_file(data_dir)

    model = Bert4RecAEW(vocab_size=len(authors) + 2,  # +2 for padding and mask
                        hidden_size=hidden_size,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        max_len=max_len + 1,  # +1 for padding
                        p_dropout=dropout)

    return model


def get_aepe_model(data_dir: str,
                   hidden_size: int = 768,
                   n_layers: int = 2,
                   n_heads: int = 8,
                   max_len: int = 200,
                   dropout: float = 0.1):
    authors = _get_authors_file(data_dir)
    papers = _get_papers_emb_file(data_dir)

    model = Bert4RecAEPE(vocab_size=len(authors) + 2,  # +2 for padding and mask
                         n_papers=papers["weight"].shape[0],
                         hidden_size=hidden_size,
                         n_layers=n_layers,
                         n_heads=n_heads,
                         max_len=max_len + 1,  # +1 for padding
                         p_dropout=dropout)

    return model


def get_aepew_model(data_dir: str,
                    hidden_size: int = 768,
                    n_layers: int = 2,
                    n_heads: int = 8,
                    max_len: int = 200,
                    dropout: float = 0.1):
    authors = _get_authors_file(data_dir)
    papers = _get_papers_emb_file(data_dir)

    model = Bert4RecAEPEW(vocab_size=len(authors) + 2,  # +2 for padding and mask
                          n_papers=papers["weight"].shape[0],
                          hidden_size=hidden_size,
                          n_layers=n_layers,
                          n_heads=n_heads,
                          max_len=max_len + 1,  # +1 for padding
                          p_dropout=dropout)

    return model


def get_seq_model(data_dir: str,
                  hidden_size: int = 768,
                  n_layers: int = 2,
                  n_heads: int = 8,
                  max_len: int = 200,
                  dropout: float = 0.1):
    authors = _get_authors_file(data_dir)
    papers = _get_papers_emb_file(data_dir)

    model = Bert4RecAEPESeq(vocab_size=len(authors) + 2,  # +2 for padding and mask
                            n_papers=papers["weight"].shape[0],
                            hidden_size=hidden_size,
                            n_layers=n_layers,
                            n_heads=n_heads,
                            max_len=max_len + 1,  # +1 for padding
                            p_dropout=dropout)

    return model


def get_nova_model(data_dir: str,
                   hidden_size: int = 768,
                   n_layers: int = 2,
                   n_heads: int = 8,
                   max_len: int = 200,
                   dropout: float = 0.1):

    authors = _get_authors_file(data_dir)
    papers = _get_papers_emb_file(data_dir)

    model = Bert4RecNova(vocab_size=len(authors) + 2,  # +2 for padding and mask
                         n_papers=papers["weight"].shape[0],
                         hidden_size=hidden_size,
                         n_layers=n_layers,
                         n_heads=n_heads,
                         max_len=max_len + 1,  # +1 for padding
                         p_dropout=dropout)

    return model


def get_data_loaders(data_dir: str,
                     task: Bert4RecTask = Bert4RecTask.EXISTING,
                     sequential: bool = False,
                     bucket: bool = False,
                     batch_size: int = 200,
                     max_len: int = 200,
                     p_mlm: float = 0.2,
                     p_mask_max: float = 0.4,
                     pad_id: int = 0,
                     mask_id: int = 1,
                     ignore_index: int = -100,
                     num_workers: int = 1) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_train, data_val, data_test = _load_data_files(data_dir, task)

    dataset_train = Bert4RecDatasetTrain(data_train,
                                         max_len=max_len,
                                         p_mlm=p_mlm,
                                         p_mask_max=p_mask_max,
                                         pad_id=pad_id,
                                         mask_id=mask_id,
                                         ignore_index=ignore_index)
    dataset_validate = Bert4RecDatasetValidate(data_val,
                                               max_len=max_len,
                                               pad_id=pad_id,
                                               mask_id=mask_id,
                                               ignore_index=ignore_index)
    dataset_test = Bert4RecDatasetTest(data_test,
                                       max_len=max_len,
                                       pad_id=pad_id,
                                       mask_id=mask_id,
                                       ignore_index=ignore_index)

    if sequential:
        collate_fn = Bert4RecDataset.collate_seq
    elif bucket:
        collate_fn = Bert4RecDataset.collate
    else:
        collate_fn = Bert4RecDataset.collate_orig

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=num_workers)
    dataloader_validate = DataLoader(dataset_validate,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    return dataloader_train, dataloader_validate, dataloader_test


def get_serializer(ignore_index: int = -100, file_name: str = None):
    return PredictionSerializer(ignore_index=ignore_index, file_name=file_name)


def _get_authors_file(data_dir: str):
    data_dir = os.path.abspath(data_dir)
    authors_file = os.path.join(data_dir, "authors.json")
    assert os.path.exists(authors_file), f"authors file '{authors_file}' does not exist"

    with open(authors_file, "r") as file:
        return json.load(file)


def _get_authors_emb_file(data_dir: str):
    data_dir = os.path.abspath(data_dir)
    authors_file = os.path.join(data_dir, "author-embedding.pt")
    assert os.path.exists(authors_file), f"author embedding file '{authors_file}' does not exist"

    return torch.load(authors_file)


def _get_papers_emb_file(data_dir: str):
    data_dir = os.path.abspath(data_dir)
    papers_file = os.path.join(data_dir, "paper-embedding.pt")
    assert os.path.exists(papers_file), f"paper embedding file '{papers_file}' does not exist"

    return torch.load(papers_file)


def _load_data_files(data_dir: str, task: Bert4RecTask) -> Tuple[list, list, list]:
    # todo put this into the dataset for sequential loading
    data_dir = os.path.abspath(data_dir)
    if task == Bert4RecTask.NEW:
        train_file = os.path.join(data_dir, "train_dataset.json")
        val_file = os.path.join(data_dir, "new_val_dataset.json")
        test_file = os.path.join(data_dir, "new_test_dataset.json")
    elif task == Bert4RecTask.EXISTING:
        train_file = os.path.join(data_dir, "train_dataset.json")
        val_file = os.path.join(data_dir, "existing_val_dataset.json")
        test_file = os.path.join(data_dir, "existing_test_dataset.json")
    else:
        raise KeyError("Invalid task specified (use enum Bert4RecTask)")
    assert os.path.exists(train_file), f"dataset file '{train_file}' does not exist"
    assert os.path.exists(val_file), f"dataset file '{val_file}' does not exist"
    assert os.path.exists(test_file), f"dataset file '{test_file}' does not exist"

    with open(train_file, "r") as file:
        data_train = [json.loads(x) for x in file]
    with open(val_file, "r") as file:
        data_val = [json.loads(x) for x in file]
    with open(test_file, "r") as file:
        data_test = [json.loads(x) for x in file]

    return data_train, data_val, data_test


if __name__ == '__main__':
    dataloader_train, dataloader_val, dataloader_test = \
        get_data_loaders(data_dir=os.path.join("data", "files-n5-ai-temporal"),
                         task=Bert4RecTask.EXISTING,
                         sequential=False,
                         bucket=True,
                         batch_size=2,
                         max_len=20,
                         num_workers=1)
    for name, dataloader in [("Train", dataloader_train), ("Val", dataloader_val), ("Test", dataloader_test)]:
        print()
        print(f"{name} ;)")
        for i in dataloader:
            for k, v in i.items():
                print(f"{k}: {v.shape}, {v[-1]}")
            break

