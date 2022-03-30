import torch
import tqdm
import numpy as np
from util.factory import get_data_loaders
from util.callbacks import Evaluation


def batch_to_oracle_prediction(batch: dict, max_length: int, pad_len: int = 10):
    res = []
    for i in range(len(batch['labels'])):
        targets = [x for x in batch["labels"][i].tolist() if x != -100]
        targets = targets[:pad_len] + [-1] * (pad_len - len(targets))
        res.append(np.tile(targets, (max_length, 1)))
    return torch.tensor(np.array(res))


def run_baseline(args: dict):
    """
    :return:
    """
    ignore_index = -100
    _, _, dataloader_test = get_data_loaders(data_dir=args["data_dir"], task=args["task"],
                                             sequential=args["seq-model"] if "seq-model" in args else None,
                                             bucket=args["bucket_embedding"] if "bucket_embedding" in args else None,
                                             batch_size=args["batch_size"], max_len=args["max_len"], p_mlm=args["p_mlm"],
                                             p_mask_max=args["p_mask_max"], ignore_index=ignore_index,
                                             num_workers=args["workers"] if "workers" in args else 1)

    evaluation = Evaluation(ks=args["ks"], ignore_index=ignore_index)
    callbacks = [evaluation]

    pbar = tqdm.tqdm(enumerate(dataloader_test, start=1), total=len(dataloader_test))
    for i, batch in pbar:
        logits = batch_to_oracle_prediction(batch=batch, max_length=args['max_len'])
        for callback in callbacks:
            callback(logits, batch["labels"])

        pbar_description = f"test | batch {i:d}/{len(dataloader_test)}"
        pbar.set_description(pbar_description)

    print(f"Finished: {str(evaluation)}")


if __name__ == '__main__':
    for dataset_path in ["data/files-n5-ai-temporal"]:
        run_baseline(args={
            "data_dir": dataset_path,
            "task": "existing",  # or new
            "batch_size": 16,
            "max_len": 50,
            "p_mlm": 2e-1,
            "p_mask_max": 5e-1,
            "ks": [1, 5, 10],
        })
