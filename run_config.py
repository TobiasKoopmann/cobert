from distutils.command.config import config
import json
from enum import Enum
from operator import mod
from subprocess import run
from jinja2 import Environment, FileSystemLoader


class Model(Enum):
    OGMODEL = "og_model"
    NOVAMODEL = "nova_model"
    COBERT = "cobert_model"
    SEQ_MODEL = "seq_model"
    WEIGHTED = "weighted_embedding"


class Dataset(Enum):
    AI_SMALL = "files-n5-ai-temporal"
    AI_LARGE = "files-n10-ai-temporal"
    MED_SMALL = "files-n5-medline-temporal"
    MED_LARGE = "files-n10-medline-temporal"


class Task(Enum):
    NEW = "new"
    EXISTING = "existing"


def create_config(model: Model,
                  dataset: Dataset,
                  task: Task,
                  trials: int,
                  debug: bool = False) -> dict:
    config = {}
    config["model"] = curr_model.value.split("_")[0]
    config["dataset"] = "-".join(curr_dataset.value.split("-")[-2:])
    config["data_dir"] = f"data/{dataset.value}"
    config['study_name'] = f"{dataset.value}-{task.value}"
    config["db_path"] = f"data/journal_runs/{model.value.split('-')[0]}-{task.value}.db"
    config["task"] = task.value
    config["trials"] = trials
    config[f"{model.value}"] = "True"
    if model != model.OGMODEL:
        config["pretrained_author_embedding"] = "True"
        config["pretrained_paper_embedding"] = "True"
        config["paper_embedding"] = "True"
        # config["weighted_embedding"] = False
    if debug:
        config["debug"] = "True"
    listed_config = []
    for k, v in config.items():
        listed_config.append({'key': k, 'value': v})
    return listed_config


if __name__ == '__main__':
    for curr_model, curr_dataset, curr_task in [
        (Model.NOVAMODEL, Dataset.AI_SMALL, Task.EXISTING),
        (Model.COBERT, Dataset.AI_SMALL, Task.EXISTING),
        (Model.OGMODEL, Dataset.AI_SMALL, Task.EXISTING),
        (Model.SEQ_MODEL, Dataset.AI_SMALL, Task.EXISTING),
        (Model.NOVAMODEL, Dataset.AI_SMALL, Task.NEW),
        (Model.COBERT, Dataset.AI_SMALL, Task.NEW),
        (Model.OGMODEL, Dataset.AI_SMALL, Task.NEW),
        (Model.SEQ_MODEL, Dataset.AI_SMALL, Task.NEW),
    ]:
        config = create_config(model=curr_model, dataset=curr_dataset, task=curr_task, trials=20)
        file_loader = FileSystemLoader('kubernetes')
        env = Environment(loader=file_loader)
        template = env.get_template('template.yml')
        tags = f"{curr_model.value.split('_')[0]}, {'-'.join(curr_dataset.value.split('-')[-2:])}, {curr_task.value}"
        template.stream(model=curr_model.value.split("_")[0], dataset="-".join(curr_dataset.value.split("-")[-2:]), task=curr_task.value, configs=config, tags=tags).dump('tmp.yml')
        run(["./run.sh", curr_model.value.split("_")[0], "-".join(curr_dataset.value.split("-")[-2:]), curr_task.value ])
