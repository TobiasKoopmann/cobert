import os
import json
import torch
import tqdm
import copy
import random
import multiprocessing
import numpy as np
from collections import defaultdict
from functools import partial
from util.PaperEmbeddings import calculate_embeddings, init


def create_dataset(papers_path_old: str, paper_embedding_path: str, data_dir: str = "./data/", dataset: str = "medline",
                   min_co_authors: int = 5, embedding_dim: int = 768, n_negative_samples: int = 100):
    print("Creating Embeddings. ")
    parse_as_list(papers_path_old, paper_embedding_path)
    print("Load file. ")
    papers = load_data_file(paper_embedding_path, saved_as_list=False)
    save_to = os.path.join(data_dir, f"files-n{min_co_authors}-{dataset}")
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    print("Filter ... ")
    papers_by_author, valid_authors = filter_authors(papers, min_co_authors)
    print("Create ranking instances ...")
    ranking_instances, paper_embeddings, paper_ids = get_ranking_instances(papers_by_author, valid_authors,
                                                                           embedding_dim)
    with open(os.path.join(save_to, "ranking-dataset.json"), "w") as file:
        json.dump(ranking_instances, file)
    print("Create prediction instances ...")

    predict_train, predict_validate, predict_test = get_predict_instances(papers_by_author, valid_authors, paper_ids)
    with open(os.path.join(save_to, "predict-dataset-train.json"), "w") as file:
        json.dump(predict_train, file)
    with open(os.path.join(save_to, "predict-dataset-validate.json"), "w") as file:
        json.dump(predict_validate, file)
    with open(os.path.join(save_to, "predict-dataset-test.json"), "w") as file:
        json.dump(predict_test, file)

    negative_samples = create_negative_samples(ranking_instances, valid_authors, n_negative_samples)
    with open(os.path.join(save_to, "negative-samples.json"), "w") as file:
        json.dump(negative_samples, file)

    author_embedding = get_author_embedding(papers, valid_authors, embedding_dim)
    with open(os.path.join(save_to, "authors.json"), "w") as file:
        json.dump(valid_authors, file)

    paper_embedding = torch.nn.Embedding(max(paper_embeddings.keys()) + 1, embedding_dim, padding_idx=0)
    for k, v in paper_embeddings.items():
        paper_embedding.weight[k, :] = torch.tensor(v)
    torch.save(paper_embedding.state_dict(), os.path.join(save_to, f"paper-embedding.pt"))

    torch.save(author_embedding.state_dict(), os.path.join(save_to, f"author-embedding.pt"))


def parse_as_list(old_path: str, new_path: str) -> None:
    """
    Creating Paper embeddings.
    :param old_path:
    :param new_path:
    :return:
    """
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    with open(old_path, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
    tok, mod, dev = init()
    tmp = [json.loads(x) for x in tmp]
    batches = int(len(tmp) / 128) + 1
    pbar = tqdm.tqdm(total=batches, desc="Creating embeddings. ")
    with open(new_path, 'w', encoding='utf-8') as f:
        for batch in batch(tmp, 128):
            abstracts = [curr['abstract'] for curr in batch]
            embeddings = calculate_embeddings(tok, mod, dev, abstracts).astype(float)
            for curr_dict, embedding in zip(batch, embeddings):
                curr_dict['abstract_embedding'] = list(embedding)
                json.dump(curr_dict, f)
                f.write("\n")
            pbar.update(1)
    pbar.close()


def load_data_file(file_path: str, saved_as_list: bool = True) -> list:
    """
    :param file_path:
    :param saved_as_list:
    :return:
    """
    print("load papers file")
    with open(file_path, "r", encoding='utf-8') as f:
        if saved_as_list:
            papers = json.load(f)
        else:
            papers = [json.loads(x) for x in f.readlines()]

    m = 0
    for paper in papers:
        if "ss_id" not in paper:
            paper["ss_id"] = m
            m += 1

    return papers


def filter_authors(papers: list, min_co_authors: int) -> (dict, dict):
    """
    :param papers:
    :param min_co_authors:
    :return: dict: {author: [papers]}, dict: {author: id}
    """
    print("filter valid authors")
    papers_by_author = defaultdict(list)

    for paper in papers:
        for author in paper["author"]:
            papers_by_author[author].append(paper)

    valid_authors = set(papers_by_author.keys())

    # remove authors with less than n coauthors iteratively until convergence
    authors_left = None
    while authors_left != len(valid_authors):
        authors_left = len(valid_authors)
        for author in valid_authors.copy():
            sorted_papers = sorted(papers_by_author[author], key=lambda x: int(x["year"]))
            co_authors = [i for i, paper in enumerate(sorted_papers)
                          for a in paper["author"] if a in valid_authors and author != a]
            if len(co_authors) == 0:
                valid_authors.remove(author)
                continue
            index = co_authors.index(max(co_authors))
            # there must be at least three authors before the last paper, two for unmasked and mask and one for validation
            if len(co_authors[:index]) < 3 or len(co_authors) < min_co_authors:
                valid_authors.remove(author)

    # start=2 to reserve 0 for padding and 1 for mask
    valid_authors = {k: i for i, k in enumerate(valid_authors, start=2)}

    return papers_by_author, valid_authors


def get_ranking_instances(papers_by_author: dict, valid_authors: dict, embedding_dim: int) -> (list, dict, dict):
    """
    :param papers_by_author:
    :param valid_authors:
    :param embedding_dim:
    :return:
    """
    instances = []

    paper_embeddings = {}
    paper_ids = {}

    for author in tqdm.tqdm(valid_authors, desc="create dataset"):
        instance = {
            "author": author,
            "co_authors": [],
            "paper_ids": []
        }

        for paper in papers_by_author[author]:
            paper_ids.setdefault(paper["ss_id"], len(paper_ids) + 2)  # +2 to reserve 0 for padding and 1 for mask
            paper_id = paper_ids[paper["ss_id"]]
            if paper_id not in paper_embeddings:
                if "abstract_embedding" in paper:
                    paper_embeddings[paper_id] = paper["abstract_embedding"]
                else:
                    paper_embeddings[paper_id] = [0] * embedding_dim

        sorted_papers = sorted(papers_by_author[author], key=lambda x: int(x["year"]))

        for i, paper in enumerate(sorted_papers):
            for co_author in paper["author"]:
                if co_author in valid_authors and author != co_author:
                    instance["co_authors"].append(valid_authors[co_author])
                    instance["paper_ids"].append(paper_ids[paper["ss_id"]])

        instances.append(instance)

    random.shuffle(instances)

    return instances, paper_embeddings, paper_ids


def get_predict_instance(author, sequence, paper_ids, masked_ids):
    return {
        "author": author,
        "co_authors": sequence,
        "paper_ids": paper_ids,
        "masked_ids": masked_ids
    }


def get_predict_instances(papers_by_author, valid_authors, paper_ids):
    train, val, test = [], [], []
    for author in valid_authors:
        sorted_papers = sorted(papers_by_author[author], key=lambda x: int(x["year"]))

        sequence, papers = [], []
        for index, paper in enumerate(sorted_papers):
            for co_author in paper["author"]:
                if co_author in valid_authors and author != co_author:
                    sequence.append(valid_authors[co_author])  # [1,2,5,6 ...
                    papers.append(paper_ids[paper["ss_id"]])  # [3, 3, 3, 7, 7,6 ...

        for i in reversed(range(len(sequence))):
            if sequence[i] not in sequence[:i] and sequence[i] not in sequence[i + 1:]:
                test.append(copy.deepcopy(get_predict_instance(author, sequence, papers, [i])))
                del sequence[i]
                del papers[i]
                break

        for i in reversed(range(len(sequence))):
            if sequence[i] not in sequence[:i] and sequence[i] not in sequence[i + 1:]:
                val.append(copy.deepcopy(get_predict_instance(author, sequence, papers, [i])))
                del sequence[i]
                del papers[i]
                break

        train.append(get_predict_instance(author, sequence, papers, []))
    random.shuffle(train)
    return train, val, test


def get_author_distribution(dataset, normalize: bool = True):
    distribution = {}
    for instance in dataset:
        for author in instance["co_authors"]:
            distribution.setdefault(author, 0)
            distribution[author] += 1

    if normalize:
        n_total = sum(distribution.values())
        return {k: v / n_total for k, v in distribution.items()}
    else:
        return distribution


def sample_negative(choices, probabilities, n, instance):
    samples = np.random.choice(choices, p=probabilities, size=n).tolist()
    for i in range(n):
        while samples[i] in instance["co_authors"]:
            samples[i] = np.random.choice(choices, p=probabilities).item()
    return instance["author"], samples


def create_negative_samples(dataset, authors, n_samples: int = 100):
    authors_distribution = get_author_distribution(dataset)
    choices = list(authors_distribution.keys())
    probabilities = list(authors_distribution.values())

    negative_samples = {}

    map_func = partial(sample_negative, choices, probabilities, n_samples)
    with multiprocessing.Pool() as pool:
        for author, samples in tqdm.tqdm(pool.imap_unordered(map_func, dataset), total=len(dataset),
                                         desc="negative sampling"):
            author_id = authors[author]
            negative_samples[author_id] = samples

    return negative_samples


def get_author_embedding(papers, valid_authors, embedding_dim):
    d = 768
    author_embeddings = {}
    for author in valid_authors.values():
        author_embeddings.setdefault(author, {"embedding": np.zeros(d), "n": 0})
    for paper in tqdm.tqdm(papers, desc="create author embedding"):
        if "abstract_embedding" not in paper:
            continue
        for author in paper["author"]:
            if author not in valid_authors:
                continue
            aid = valid_authors[author]
            author_embeddings[aid]["embedding"] += np.array(paper["abstract_embedding"])
            author_embeddings[aid]["n"] += 1
    for author in author_embeddings:
        author_embeddings[author] = (
                author_embeddings[author]["embedding"] / max(author_embeddings[author]["n"], 1)).tolist()

    embedding = torch.nn.Embedding(len(valid_authors) + 2, embedding_dim, padding_idx=0)
    for author in author_embeddings:
        embedding.weight[author, :] = torch.tensor(author_embeddings[author])
    return embedding


if __name__ == "__main__":
    create_dataset(papers_path_old="./data/ai_dataset_test.json",
                   paper_embedding_path="./data/papers-fixed-embedding-768-ai-test.json",
                   dataset="ai-test", min_co_authors=5)
