import json
import os.path


def load_results_json(input_path: str) -> list:
    results = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def add_stuff(results: list) -> list:
    for config in results:
        for value in ["og-model", "seq-model", "nova-model", "cobert-model",
                      "weighted_embedding", "paper_embedding", "author_embedding",
                      "pretrained_paper_embedding", "pretrained_author_embedding"]:
            if value not in config:
                config[value] = False
    return results


if __name__ == '__main__':
    results = load_results_json(os.path.join("data", "journal_runs", "results.json"))
    results = add_stuff(results=results)
    with open(os.path.join("data", "journal_runs", "results_new.json"), 'w', encoding='utf-8') as f:
        for line in results:
            json.dump(line, f)
            f.write("\n")
    print("Finished. ")