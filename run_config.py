import json
import uuid
from enum import Enum
from subprocess import run


class Model(Enum):
    OGMODEL = "og"
    NOVAMODEL = "nova"
    COBERT = "cobert"
    SEQ_MODEL = "seq"


class Dataset(Enum):
    AI_SMALL = "files-n5-ai-temporal"
    AI_LARGE = "files-n5-ai-temporal"
    MED_SMALL = "files-n5-medline-temporal"
    MED_LARGE = "files-n10-medline-temporal"


class Task(Enum):
    NEW = "new"
    EXISTING = "existing"


def create_config(model: Model, dataset: Dataset, task: Task, in_path: str = "config.json", out_path: str = "tmp.json") -> None:
    config = json.load(open(in_path, "r", encoding="utf-8"))
    config["data_dir"] = f"data/{dataset.value}"
    config['study_name'] = f"{dataset.value}-{task.value}"
    config["db_path"] = f"data/journal_runs/{task.value}.db"
    config["task"] = task.value
    config[f"{model.value}-model"] = True
    if model != model.OGMODEL:
        config["pretrained_author_embedding"] = True
        config["pretrained_paper_embedding"] = True
        config["paper_embedding"] = True
    json.dump(config, open(out_path, "w", encoding="utf-8"))


if __name__ == '__main__':
    for curr_model, curr_dataset, curr_task in [
        (Model.NOVAMODEL, Dataset.AI_SMALL, Task.NEW),
        (Model.COBERT, Dataset.AI_SMALL, Task.NEW),
        (Model.OGMODEL, Dataset.AI_SMALL, Task.NEW),
        (Model.SEQ_MODEL, Dataset.AI_SMALL, Task.NEW),
    ]:
        print(f"Creating {curr_model.value} for {curr_dataset.value}. ")
        create_config(model=curr_model, dataset=curr_dataset, task=curr_task)
        run(["./run.sh", curr_model.value, curr_dataset.value])
