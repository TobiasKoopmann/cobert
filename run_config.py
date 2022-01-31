import json
import uuid
from enum import Enum
from subprocess import run


class Model(Enum):
    OGMODEL = "og-model"
    NOVAMODEL = "nova-model"
    COBERT = "cobert"
    SEQ_MODEL = "seq-model"


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
    curr_id = str(uuid.uuid4())
    config["data_dir"] = f"data/{dataset.value}"
    config["id"] = curr_id
    config["run_dir"] = f"data/journal_runs/{curr_id}/"
    config["save_predictions_file"] = f"data/journal_runs/{curr_id}/predictions"
    config["task"] = task.value
    config[model.value] = True
    if model != model.OGMODEL:
        config["pretrained_author_embedding"] = True
        config["pretrained_paper_embedding"] = True
        config["paper_embedding"] = True
    json.dump(config, open(out_path, "w", encoding="utf-8"))


if __name__ == '__main__':
    for curr_model, curr_dataset, curr_task in [
        (Model.NOVAMODEL, Dataset.AI_SMALL, Task.NEW),
        (Model.OGMODEL, Dataset.AI_SMALL, Task.NEW),
        (Model.SEQ_MODEL, Dataset.AI_SMALL, Task.NEW),
    ]:
        print(f"Creating {curr_model.value} for {curr_dataset.value}. ")
        create_config(model=curr_model, dataset=curr_dataset, task=curr_task)
        run(["./run.sh", f"{curr_model.value}-{curr_dataset.value}"])
