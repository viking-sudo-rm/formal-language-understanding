from copy import deepcopy
import os
import numpy as np
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from collections import defaultdict
import matplotlib.pyplot as plt

from allennlp.models.archival import load_archive
from allennlp.common.util import prepare_environment
from allennlp.predictors import Predictor
from allennlp_models.lm.dataset_readers import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", "-M", action="append")
    parser.add_argument("--eval_path", type=str, default="data/eval.tsv")
    parser.add_argument("--false_only", action="store_true")
    return parser.parse_args()


def contrast(sentence):
    """Modify the sentence by swapping """
    value = eval(sentence[-1])
    new_sent = deepcopy(sentence)
    new_sent[-1] = str(not value)
    return new_sent


def get_predictor(path):
    archive = load_archive(os.path.join(path, "model.tar.gz"), cuda_device=-1)
    config = deepcopy(archive.config)
    prepare_environment(config)
    model = archive.model
    model.eval()
    dataset_reader = archive.validation_dataset_reader
    predictor = Predictor(model, dataset_reader)
    return predictor


def _featurize(sentence, predictor):
    dataset_reader = predictor._dataset_reader
    instance = dataset_reader.text_to_instance(sentence)
    output = predictor.predict_instance(instance)
    # Take index 1 for the actual word in this one-word sentence.
    return output["lm_embeddings"][1]


def featurize(sents1, sents2, predictor):
    features1 = np.array([_featurize(s, predictor) for s in sents1])
    features2 = np.array([_featurize(s, predictor) for s in sents2])
    return np.concatenate([features1, features2], axis=1)


def get_data(path):
    with open(path) as fh:
        tsv_contents = [line.strip().split("\t") for line in fh.readlines()]
    sents1, sents2, labels = zip(*tsv_contents)
    labels = np.array([1 if l == "True" else 0 for l in labels])
    return sents1, sents2, labels


def split(data):
    split_idx = len(data) // 2
    return data[:split_idx], data[split_idx:]


args = parse_args()
predictors = [get_predictor(model) for model in args.model]
sents1, sents2, labels = get_data(args.eval_path)
accs = defaultdict(list)

for _ in range(20):
    sents1, sents2, labels = shuffle(sents1, sents2, labels)
    train_sents1, test_sents1 = split(sents1)
    train_sents2, test_sents2 = split(sents2)
    train_labels, test_labels = split(labels)
    for model_name in args.model:
        predictor = get_predictor(model_name)
        train_features = featurize(train_sents1, train_sents2, predictor)
        test_features = featurize(test_sents1, test_sents2, predictor)
        clf = LogisticRegression().fit(train_features, train_labels)
        preds = clf.predict(test_features)
        acc = (preds == test_labels).sum()
        accs[model_name].append(acc)
    true_acc = (test_labels == 1).sum()
    false_acc = (test_labels == 0).sum()
    maj_acc = max(true_acc, false_acc)
    accs["baseline"].append(maj_acc)

plt.figure()
for model, data in accs.items():
    print(f"{model} mean: {np.mean(data)}")
    plt.hist(data, label=model, bins=list(range(min(data), max(data) + 1)), alpha=.2)
plt.legend()
plt.tight_layout()
plt.show()

# predictions = []
# ground_truths = []
# probabilities = []
# while instances:
#     instance1 = instances.pop(0)
#     instance2 = instances.pop(0)
#     label1 = labels.pop(0) == "True"
#     labels.pop(0)
#     loss1 = predictor.predict_instance(instance1)["loss"]
#     loss2 = predictor.predict_instance(instance2)["loss"]
#     if args.false_only:
#         loss1 = 1.
#         loss2 = 0.
#     predicted1 = loss1 < loss2
#     cond_prob1 = np.exp(loss1) / (np.exp(loss1) + np.exp(loss2))
#     predictions.append(0 if predicted1 else 1)
#     ground_truths.append(0 if label1 else 1)
#     probabilities.append([cond_prob1, 1 - cond_prob1])

# predictions = np.array(predictions)
# ground_truths = np.array(ground_truths)
# probabilities = np.array(probabilities)

# accuracy = np.sum(predictions == ground_truths) / len(predictions)
# cross_ent = np.sum(-np.log(probabilities)[ground_truths]) / len(predictions)

# print("Accuracy", accuracy)
# print("Cross ent", cross_ent)
