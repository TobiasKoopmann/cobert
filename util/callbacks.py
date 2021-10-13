import os
import abc
import json
import numpy as np


class Callback(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, predictions, labels):
        pass


class Evaluation(Callback):
    def __init__(self, negative_samples, ks=(1, 5, 10), ignore_index: int = -100):
        super().__init__()
        self.ks = ks
        self.negative_samples = {int(k): v for k, v in negative_samples.items()}
        self.ignore_index = ignore_index
        self.evaluation = None
        self.reset()

    def __call__(self, predictions, labels):  # predictions: torch [batch, 50, 35115], labels: [batch, 50]
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == self.ignore_index:  # [ignore, ignore, ..., mask]
                    continue
                candidate = labels[i, j].item()  # integer
                samples = self.negative_samples[candidate] + [candidate]
                sample_predictions = predictions[i, j][samples].tolist()
                ranked_samples = list(sorted(zip(samples, sample_predictions), key=lambda x: x[1], reverse=True))  # list of id, logit
                self.evaluation["n"] += 1
                rank = 0
                for index, sample in enumerate(ranked_samples):
                    if sample[0] == candidate:
                        rank = index
                        break
                for k in self.ks:
                    if rank < k:
                        self.evaluation["ndcg"][k] += 1 / np.log2(rank + 2)
                        self.evaluation["hit"][k] += 1

    def __str__(self):
        return " ".join(
            f"{key}@{k}={self.evaluation[key][k] / self.evaluation['n']:.5f}" for key in ("ndcg", "hit") for k in
            self.evaluation[key])

    def reset(self):
        self.evaluation = {"ndcg": {k: 0 for k in self.ks},
                           "hit": {k: 0 for k in self.ks},
                           "n": 0}

    def get_metric(self, metric: str):
        if metric in self.evaluation:
            return [(k, self.evaluation[metric][k] / self.evaluation['n']) for k in self.evaluation[metric]]


class PredictionSerializer(Callback):
    def __init__(self,
                 file_name: str,
                 ignore_index: int = -100):
        super().__init__()
        self.predictions = []
        self.labels = []
        self.ignore_index = ignore_index
        parent_dir = os.path.dirname(file_name)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.file = open(file_name, "w")
        self.file.write("Prediction\tLabel\n")

    def __call__(self, predictions, labels):
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] != self.ignore_index:
                    self.predictions.append(np.argsort(predictions[i, j].cpu()).tolist()[:100])
                    self.labels.append(labels[i, j].item())
        for p, l in zip(self.predictions, self.labels):
            self.file.write(",".join([str(x) for x in p]) + "\t" + str(l) + "\n")
        self.predictions, self.labels = [], []

    def serialize(self, file_path: str):
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        with open(file_path, "w") as file:
            json.dump({
                "predictions": self.predictions,
                "labels": self.labels,
            }, file)

        self.predictions, self.labels = [], []
