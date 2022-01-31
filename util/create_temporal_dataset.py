import torch
import os
import copy
import random
import tqdm
import json
import numpy as np
from collections import defaultdict

from util.PaperEmbeddings import calculate_embeddings, init


def create_temporal_dataset(orig_path: str,
                            dataset: str,
                            out_dir: str,
                            embedding_dim: int = 768,
                            min_co_authors: int = 5,
                            val_split_year: int = 2018,
                            test_split_year: int = 2019,
                            create_embeddings: bool = False) -> None:
    print("Load file. ")
    papers_by_author, authors = load_dataset_sequentially(file_path=orig_path, min_co_authors=min_co_authors)
    train, existing_test, existing_val, new_test, new_val, _, paper_ids = \
        create_test_train_data(papers_by_author=papers_by_author, authors=authors, embedding_dim=embedding_dim,
                               split_years=[val_split_year, test_split_year])

    save_to = os.path.join(out_dir, f"files-n{min_co_authors}-{dataset}")

    if not os.path.exists(save_to):
        os.makedirs(save_to)
    if create_embeddings:
        print("Creating Embeddings. ")
        create_author_and_paper_embeddings(papers_by_author=papers_by_author,
                                           save_to=save_to,
                                           paper_ids=paper_ids,
                                           authors=authors,
                                           train_year=val_split_year)

    print(f"Authors: {len(authors)}, Train: {len(train)}, "
          f"Existing Val: {len(existing_val)}, existing test: {len(existing_test)} & "
          f"New Val: {len(new_val)}, new test: {len(new_test)}")
    json.dump(authors, open(os.path.join(save_to, "authors.json"), "w", encoding="utf-8"))
    for name, dataset in [("train_dataset.json", train),
                          ("new_test_dataset.json", new_test), ("new_val_dataset.json", new_val),
                          ("existing_test_dataset.json", existing_test), ("existing_val_dataset.json", existing_val)]:
        print(f"Saving to {name}")
        with open(os.path.join(save_to, name), 'w', encoding='utf-8') as f:
            for line in dataset:
                json.dump(line, f)
                f.write("\n")


def load_dataset_sequentially(file_path: str, min_co_authors: int) -> (dict, dict):
    """

    :param file_path:
    :param min_co_authors:
    :return:
    """
    m = 0
    authors = set()
    papers_by_author = defaultdict(list)

    with open(file_path, "r", encoding='utf-8') as f:
        for line in tqdm.tqdm(f, desc="Creating author list"):
            paper = json.loads(line)
            if "ss_id" not in paper:
                paper["ss_id"] = m
                m += 1
            authors.update(paper['author'])
            for author in paper["author"]:
                papers_by_author[author].append(paper)

    for author in tqdm.tqdm(authors.copy(), desc="Filtering min occurences"):
        sorted_papers = sorted(papers_by_author[author], key=lambda x: int(x["year"]))
        co_authors = [i for i, paper in enumerate(sorted_papers) for a in paper["author"] if a in authors and author != a]
        if len(co_authors) == 0:
            authors.remove(author)
            continue
        index = co_authors.index(max(co_authors))
        # there must be at least three authors before the last paper, two for unmasked and mask and one for validation
        if len(co_authors[:index]) < 3 or len(co_authors) < min_co_authors:
            authors.remove(author)

    print(f"Having {len(authors)} authors left after filtering. ")
    # start=2 to reserve 0 for padding and 1 for mask
    valid_authors = {k: i for i, k in enumerate(authors, start=2)}

    return papers_by_author, valid_authors


def create_test_train_data(papers_by_author: dict, authors: dict, embedding_dim: int, split_years: list) \
        -> (list, list, list, list, list, list, list):
    """
    :param papers_by_author:
    :param authors:
    :param split_years:
    :return:
    train_instances, existing_train_instances, existing_val_instances, new_train_instances, new_val_instances: list of instances
    paper_embeddings: dict {int id: vector of 768}
    paper_ids: dict {int id: ss_id (some hash)}
    """
    train_instances, existing_test_instances, existing_val_instances, new_test_instances, new_val_instances = \
        [], [], [], [], []
    paper_embeddings = {}
    paper_ids = {}
    year_analysis = defaultdict(int)

    for author in tqdm.tqdm(authors, desc="create dataset"):
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

        train_filtered = list(filter(lambda x: int(x['year']) <= split_years[0], sorted_papers))

        for paper in train_filtered:
            for co_author in paper["author"]:
                if co_author in authors and author != co_author:
                    instance["co_authors"].append(authors[co_author])
                    instance["paper_ids"].append(paper_ids[paper["ss_id"]])
        if len(set(instance["paper_ids"])) > 2 and len(instance["co_authors"]) > 2:
            train_instances.append(copy.deepcopy(instance))
        else:
            continue

        if len(sorted_papers) != len(train_filtered):
            train_paper = list(filter(lambda x: int(x['year']) > split_years[0], sorted_papers))
            instance["masked_ids"] = []
            for paper in train_paper:
                fixed_instance = copy.deepcopy(instance)
                for co_author in paper['author']:
                    if co_author in authors and author != co_author:
                        curr_instance = copy.deepcopy(fixed_instance)
                        is_existing = True if authors[co_author] in instance["co_authors"] else False
                        curr_instance["masked_ids"].append(len(instance["co_authors"]))
                        curr_instance["co_authors"].append(authors[co_author])
                        curr_instance["paper_ids"].append(paper_ids[paper["ss_id"]])
                        instance["co_authors"].append(authors[co_author])
                        instance["paper_ids"].append(paper_ids[paper["ss_id"]])
                        if int(paper['year']) > split_years[1]:
                            if is_existing:
                                existing_test_instances.append(curr_instance)
                            else:
                                new_test_instances.append(curr_instance)
                        else:
                            if is_existing:
                                existing_val_instances.append(curr_instance)
                            else:
                                new_val_instances.append(curr_instance)
                year_analysis[int(paper['year'])] += 1
    print(f"Year analysis: {year_analysis}")

    random.shuffle(train_instances)
    random.shuffle(existing_test_instances)
    random.shuffle(new_test_instances)
    random.shuffle(existing_val_instances)
    random.shuffle(new_val_instances)

    return train_instances, existing_test_instances, existing_val_instances, new_test_instances, new_val_instances, \
           paper_embeddings, paper_ids


def create_author_and_paper_embeddings(papers_by_author: dict, save_to: str,
                                       paper_ids: dict, authors: dict,
                                       train_year: int, embedding_dim: int = 768):
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    tok, mod, dev = init()
    paper_abstracts = {}
    no_abstract_counter = set()
    for _, papers in papers_by_author.items():
        for paper in papers:
            if "abstract" in paper and int(paper['year']) <= train_year and paper["ss_id"] in paper_ids and paper_ids[paper["ss_id"]] not in paper_abstracts:
                paper_abstracts[paper_ids[paper['ss_id']]] = paper['abstract']
            if "abstract" not in paper:
                no_abstract_counter.add(paper['ss_id'])
    print(f"Having {len(no_abstract_counter)} paper without abstracts. ")
    batches = int(len(paper_abstracts) / 128) + 1
    pbar = tqdm.tqdm(total=batches, desc="Creating embeddings. ")
    paper_embeddings = {}
    for batch in batch(list(paper_abstracts.items()), 128):
        abstracts = [curr[1] for curr in batch]
        embeddings = calculate_embeddings(tok, mod, dev, abstracts).astype(float)
        for curr_batch, embedding in zip(batch, embeddings):
            paper_embeddings[curr_batch[0]] = list(embedding)
        pbar.update(1)
    pbar.close()
    paper_embedding = torch.nn.Embedding(max(paper_embeddings.keys()) + 1, embedding_dim, padding_idx=0)
    with torch.no_grad():
        for k, v in paper_embeddings.items():
            paper_embedding.weight[k, :] = torch.FloatTensor(v)
    torch.save(paper_embedding.state_dict(), os.path.join(save_to, f"paper-embedding.pt"))
    author_embeddings = {}
    for author, author_id in tqdm.tqdm(authors.items(), desc="Create author embeddings. "):
        curr_embeddings = [paper_embeddings[paper_ids[paper['ss_id']]] for paper in papers_by_author[author]
                           if int(paper['year']) <= train_year and paper['ss_id'] in paper_ids
                           and paper_ids[paper['ss_id']] in paper_embeddings]
        if len(curr_embeddings) > 0:
            author_embeddings[author_id] = np.array(curr_embeddings).mean(axis=0)
    author_embedding = torch.nn.Embedding(len(authors) + 2, embedding_dim, padding_idx=0)
    with torch.no_grad():
        for k, v in author_embeddings.items():
            author_embedding.weight[k, :] = torch.FloatTensor(v)
    torch.save(author_embedding.state_dict(), os.path.join(save_to, f"author-embedding.pt"))


if __name__ == '__main__':
    # datasets are ai_dataset.json, ai_community_dataset.json
    create_temporal_dataset(orig_path="./data/ai_dataset.json", dataset="ai-temporal", out_dir="./data",
                            create_embeddings=True)
    create_temporal_dataset(orig_path="./data/ai_community_dataset.json", dataset="ai-community-temporal", out_dir="./data",
                            create_embeddings=True)
    # create_temporal_dataset(orig_path="./data/medline.json", dataset="medline_temporal-n5", out_dir="./data", create_embeddings=True)
