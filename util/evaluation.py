import os
import json
import numpy as np
import pandas as pd

save_as_csv = False
print_latex = True


def print_scores(model_desc: dict) -> None:
    return " ".join(
        [f"{key}@{k}={model_desc[key][i][1]}" for key in ("hit", "ndcg") for i, k in enumerate(model_desc['ks'])])


def print_mean(models: list) -> str:
    ret = ""
    for key in ("hit", "ndcg"):
        for i, k in enumerate([1, 5, 10]):
            metric = 0
            for m in models:
                metric += m[key][i][1]
            metric /= len(models)
            ret += f"{key}@{k}={metric} "
    return ret


def get_mean_and_std(models: list, std: bool = True, get_top_5: bool = True) -> dict:
    if get_top_5:
        models = models[:5]
    mean_calc = {
        'hit': [],
        'ndcg': []
    }
    k_values = np.array(models[0]['hit'])[:, 0]
    for model in models:
        mean_calc['hit'].append(model['hit'])
        mean_calc['ndcg'].append(model['ndcg'])
    for metric in ['hit', 'ndcg']:
        metric_array = np.array(mean_calc[metric])
        means = np.mean(metric_array[:, :, 1], axis=0)
        if not std:
            mean_calc[metric] = [[idx, str(round(mean * 100, 1))] for idx, mean in zip(k_values, means)]
        else:
            std_devs = np.std(metric_array[:, :, 1], axis=0)
            mean_calc[metric] = [[idx, str(round(mean * 100, 1)) + "±" + str(round(std * 100, 1))]
                                 for idx, mean, std in zip(k_values, means, std_devs)]
    return mean_calc


def get_eval_dict(file_path: str = "data/runs/results.json",
                  sort_key: str = "val_hit",
                  data_dirs: list = ("data/files-n10", "data/files-n5"),
                  verbose: int = 1,
                  std_dev: bool = True) -> dict:
    def check(curr_dict: dict) -> bool:
        return curr_dict['task'] == task and curr_dict['data_dir'] == data_dir

    with open(file_path, "r", encoding='utf-8') as f:
        tmp = f.readlines()
    tmp = [json.loads(x) for x in tmp]
    eval_dict = {}
    for data_dir in data_dirs:
        eval_dict[data_dir] = {}
        for task in ["ranking", "predict"]:
            if verbose:
               print("-" * 30 + data_dir + " " + task + "-" * 30)
            curr_best_models = {}
            curr = [x for x in tmp if check(x)]
            # Baselines
            for baseline in ["majority", "author_majority", "graph"]:
                baseline_config = [x for x in curr if check(x) and "model" in x and baseline in x["model"]]
                if verbose:
                    print(baseline.replace("_", " ").title() + " " + str(len(baseline_config)))
                if len(baseline_config) > 0:
                    if verbose:
                        print(print_mean(baseline_config[:5]))
                    mean = get_mean_and_std(baseline_config, std=std_dev)
                    curr_best_models[baseline] = mean
                else:
                    curr_best_models[baseline] = {"hit": [[1, "---"], [5, "---"], [10, "---"]],
                                                  "ndcg": [[1, "---"], [5, "---"], [10, "---"]]}
            if verbose:
                print("Having ", len(curr), "models. Baselines are ", curr_best_models.items())
            # other models
            for model_type in ["og_model", "bucket_embedding", "paper_embedding", "pretrained_author_embedding",
                               "paper_author_embedding", "weighted_embedding", "seq_model"]:
                if model_type == "seq_model":
                    current = sorted([x for x in curr if model_type in x and x[model_type]],
                                     key=lambda x: x[sort_key][0][1], reverse=True)
                elif model_type == "weighted_embedding":
                    current = sorted([x for x in curr if model_type in x and x[model_type] and not x["seq_model"]],
                                     key=lambda x: x[sort_key][0][1], reverse=True)
                elif model_type == "paper_author_embedding":
                    current = sorted([x for x in curr if "og_model" in x and x["pretrained_author_embedding"] and
                                      x["paper_embedding"] and not x["seq_model"] and not x["weighted_embedding"]],
                                     key=lambda x: x[sort_key][0][1], reverse=True)
                elif model_type == "pretrained_author_embedding":
                    current = sorted([x for x in curr if "og_model" in x and x["pretrained_author_embedding"]
                                      and not x["paper_embedding"] and not x["seq_model"]
                                      and not x["weighted_embedding"]],
                                     key=lambda x: x[sort_key][0][1], reverse=True)
                elif model_type == "paper_embedding":
                    current = sorted([x for x in curr if "og_model" in x and x["paper_embedding"]
                                      and not x["pretrained_author_embedding"] and not x["seq_model"]
                                      and not x["weighted_embedding"]],
                                     key=lambda x: x[sort_key][0][1], reverse=True)
                elif model_type == "bucket_embedding":
                    current = sorted([x for x in curr if model_type in x and x[model_type]],
                                     key=lambda x: x[sort_key][0][1], reverse=True)
                else:  # orig
                    current = sorted([x for x in curr if model_type in x and x[model_type] and not x["seq_model"]
                                      and not x["weighted_embedding"]], key=lambda x: x[sort_key][0][1], reverse=True)
                if verbose:
                    print(model_type.replace("_", " ").title() + " " + str(len(current)))
                if len(current) > 0:
                    if verbose:
                        print(print_mean(current[:5]))
                        for i in range(min(5, len(current))):
                            print(current[i]['id'])
                            print([(x, current[i][x]) for x in ['hit', 'ndcg', 'val_hit', 'val_ndcg']])
                    mean = get_mean_and_std(current, std=std_dev)
                    curr_best_models[model_type] = mean
                else:
                    print("Fail.")
            eval_dict[data_dir][task] = curr_best_models
    return eval_dict


def print_main_table(eval_dict: dict) -> None:
    print()
    for data_dir, tasks in eval_dict.items():
        for task_name, results in tasks.items():
            curr = {"models": ["Majority", "Majority (Author)", "Graph", "BERT4Rec", "CoBERT", "CoBERT-W", "CoBERT-Seq"]}
            curr.update({x[0]: [results['majority'][x[1]][x[2]][1],
                                results['author_majority'][x[1]][x[2]][1],
                                results['graph'][x[1]][x[2]][1],
                                results['og_model'][x[1]][x[2]][1],
                                results['paper_author_embedding'][x[1]][x[2]][1],
                                results['weighted_embedding'][x[1]][x[2]][1],
                                results['seq_model'][x[1]][x[2]][1] if 'seq_model' in results else "---"] for x in
                         [["Acc.", "hit", 0], ["Hit@5", "hit", 1], ["Hit@10", "hit", 2], ["NDCG@5", "ndcg", 1],
                          ["NDCG@10", "ndcg", 2]]})
            df = pd.DataFrame(curr)
            # print(df)
            print(df.to_latex(index=False, caption=data_dir.split("/")[1].split("-")[1] + "-" + task_name, label="tab:" + data_dir.split("/")[1].split("-")[1] + "-" + task_name))


def print_ablation_study(eval_dict: dict, models: list, dataset: str, task: str) -> None:
    curr = eval_dict[dataset][task]
    df_dict = {"models": ["No ACE, PCE & PPE", "No ACE & PCE", "No PCE", "No ACE", "CoBERT"]}
    for k, v in curr.items():
        if k in models:
            print(k, ": ", v)
    print()
    for metric in [("Acc.", 'hit', 0), ("Hit@5", "hit", 1), ("Hit@10", "hit", 2), ("NCDG@5", "ndcg", 1), ("NCDG@10", "ndcg", 2)]:
        curr_list = []
        for model in models:
            curr_list.append(curr[model][metric[1]][metric[2]][1])
        df_dict[metric[0]] = curr_list
    df = pd.DataFrame(df_dict)
    dataset_name = "PlosOne" if "medline" in dataset else "AI"
    dataset_size = "$n=5$" if "n5" in dataset else "$n=10$"
    task_name = "any" if task == "ranking" else "new"
    print(df.to_latex(index=False, caption="Ablation study based on the " + dataset_name + " dataset with " +
                                           dataset_size + " in the predicting " + task_name + " collaborator task.",
          label="tab:" + dataset_name + "-" + dataset_size.replace("$", "").replace("=", "") + "-" + task_name))


def analyse_embedding_weights(path_to_file: str) -> None:
    from collections import namedtuple
    from util.factory import get_model
    import torch
    import os
    with open(os.path.join(path_to_file, "config.json"), 'r', encoding='utf-8') as f:
        config = json.load(f)
    args = namedtuple('Struct', config.keys())(*config.values())
    model = get_model(args)
    model.load_state_dict(torch.load(os.path.join(path_to_file, "model.pt"), map_location=torch.device('cpu')))
    for name, param in model.named_parameters():
        if param.requires_grad and name == "embedding.embedding_weights":
            print(name, ":", param.data)


if __name__ == '__main__':
    print_main = True
    print_ablation = False
    analyse_weights = False
    eval_dict = get_eval_dict(data_dirs=["data/files-n5", "data/files-n10",
                                         "data/files-n5-medline", "data/files-n10-medline"],
                              verbose=not analyse_weights, std_dev=False)
    if print_main:
        print_main_table(eval_dict=eval_dict)
    if print_ablation:
        for data in ["data/files-n5", "data/files-n10", "data/files-n5-medline", "data/files-n10-medline"]:
            for task in ["ranking", "predict"]:
                print_ablation_study(eval_dict=eval_dict, dataset=data, task=task,
                                     models=["og_model", "bucket_embedding", "pretrained_author_embedding",
                                             "paper_embedding", "paper_author_embedding"])
    if analyse_weights:
        print("Weights: Token, Author, Position")
        for file in ["9c08458b1f42", "404a4045ac29", "35c9013b868a"]:  # PlosOne - AI
            analyse_embedding_weights(path_to_file=os.path.join("data", "runs", file))
