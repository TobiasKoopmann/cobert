import json
from enum import Enum
from subprocess import run


class Model(Enum):
    OGMODEL = "og-model"
    NOVAMODEL = "nova-model"
    COBERT = "cobert-model"
    SEQ_MODEL = "seq-model"
    WEIGHTED = "weighted_embedding"


class Dataset(Enum):
    AI_SMALL = "files-n5-ai-temporal"
    AI_LARGE = "files-n5-ai-temporal"
    MED_SMALL = "files-n5-medline-temporal"
    MED_LARGE = "files-n10-medline-temporal"


class Task(Enum):
    NEW = "new"
    EXISTING = "existing"


def create_config(model: Model,
                  dataset: Dataset,
                  task: Task,
                  trials: int,
                  in_path: str = "config.json",
                  out_path: str = "tmp.json",
                  debug: bool = False) -> None:
    config = json.load(open(in_path, "r", encoding="utf-8"))
    config["data_dir"] = f"data/{dataset.value}"
    config['study_name'] = f"{dataset.value}-{task.value}"
    config["db_path"] = f"data/journal_runs/{model.value.split('-')[0]}-{task.value}.db"
    config["task"] = task.value
    config["trials"] = trials
    for possible_model in [Model.SEQ_MODEL, Model.OGMODEL, Model.COBERT, Model.NOVAMODEL]:
        if possible_model == model:
            config[f"{model.value}"] = True
        else:
            config[f"{possible_model.value}"] = False
    config["pretrained_author_embedding"] = model != model.OGMODEL
    config["pretrained_paper_embedding"] = model != model.OGMODEL
    config["paper_embedding"] = model != model.OGMODEL
    config["weighted_embedding"] = False
    if debug:
        config["debug"] = True
    json.dump(config, open(out_path, "w", encoding="utf-8"))


if __name__ == '__main__':
    for curr_model, curr_dataset, curr_task in [
        # (Model.NOVAMODEL, Dataset.AI_SMALL, Task.EXISTING),
        (Model.COBERT, Dataset.AI_SMALL, Task.EXISTING),
        # (Model.OGMODEL, Dataset.AI_SMALL, Task.EXISTING),
        # (Model.SEQ_MODEL, Dataset.AI_SMALL, Task.EXISTING),
    ]:
        print(f"Creating {curr_model.value.split('-')[0]} for {curr_dataset.value}. ")
        create_config(model=curr_model, dataset=curr_dataset, task=curr_task, trials=20)
        run(["./run.sh", curr_model.value.split("-")[0], curr_dataset.value])
