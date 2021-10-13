import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        curr = json.load(f)
    return curr


def analyse_file(path: str) -> (int, int, float, float):
    """
    :param path:
    :return:
    """
    authors = load_json(os.path.join(path, "authors.json"))
    dataset = load_json(os.path.join(path, "ranking-dataset.json"))
    authors_to_paper = defaultdict(list)
    papers_to_authors = defaultdict(list)
    avg_co_authors, avg_unique_co_authors, avg_papers, total_papers = 0, 0, 0, set()
    for author in tqdm(dataset):
        for co_auth, paper in zip(author['co_authors'], author['paper_ids']):
            authors_to_paper[co_auth].append(paper)
            authors_to_paper[authors[author['author']]].append(paper)
            papers_to_authors[paper].extend([authors[author['author']], co_auth])
        total_papers.update(author['paper_ids'])
    for paper in authors_to_paper.values():
        avg_papers += len(paper)
    for author in papers_to_authors.values():
        avg_co_authors += len(author)
        avg_unique_co_authors += len(set(author))
    print("Authors: ", len(authors),
          "\tDataset: ", len(dataset),
          "\ttotal papers: ", len(total_papers),
          "\tavg co authors: ", (avg_co_authors/len(dataset)),
          "\tavg unique co authors: ", (avg_unique_co_authors/len(dataset)),
          "\tavg papers:", (avg_papers/len(dataset)))
    return len(authors), len(total_papers), (avg_co_authors/len(dataset)), (avg_papers/len(dataset))


def analyse_datasets(args: dict) -> str:
    """

    :return: latex string
    """
    results = {'datasets': ['authors', 'papers', 'avg. co-authors']}
    for dataset in args['datasets']:
        authors, papers, avg_co_authors, avg_papers = analyse_file(os.path.join(args['base_dir'], dataset))
        if "10" in dataset:
            dataset_name = "n=10"
        else:
            dataset_name = "n=5"
        results[dataset_name] = [authors, papers, avg_co_authors]
    return pd.DataFrame(results).to_latex(index=False, float_format="{:0.2f}".format, label="tab:dataset",
                                          caption=" Characteristics. $n$ is the smallest number of co-authors to be considered in the dataset. ")


def compare_train_test_val_instances(path: str):
    for dataset in ["predict-dataset-train.json", "predict-dataset-validate.json", "predict-dataset-test.json"]:
        curr = load_json(path + dataset)
        print(dataset.split("-")[-1].split(".")[0], " instances: \t", len(curr))


if __name__ == '__main__':
    print(analyse_datasets({
        'base_dir': "./data/",
        'datasets': ["files-n5", "files-n10", "files-n5-medline", "files-n10-medline"]
    }))
    # for file in ["data/files-n5/", "data/files-n10/", "data/files-n5-medline/", "data/files-n10-medline/"]:
    #     print("-" * 30 + file + "-" * 30)
    #     compare_train_test_val_instances(file)
