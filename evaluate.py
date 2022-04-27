from copy import deepcopy
import os
import numpy as np
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from allennlp.training.metrics.auc import Auc

from allennlp.models.archival import load_archive
from allennlp.common.util import prepare_environment
from allennlp.predictors import Predictor
from allennlp_models.lm.dataset_readers import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/quantifier/", help="Directory containing all models to evaluate")
    parser.add_argument("--eval_path", type=str, default="data/quantifier/eval.tsv")
    parser.add_argument("--false_only", action="store_true")
    parser.add_argument("--analysis_method", type=str, help="Options: [logistic_regression, uniform_true, independently_truthful, informative]")
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



def get_instance(sentence, predictor):
    dataset_reader = predictor._dataset_reader
    return dataset_reader.text_to_instance(sentence)


def get_lm_embeddings(sentence, predictor):
    instance = get_instance(sentence, predictor)
    output = predictor.predict_instance(instance)
    return output["lm_embeddings"]

def _featurize(sentence, predictor):
    lm_embeddings = get_lm_embeddings(sentence, predictor)
    return lm_embeddings[1]  # Take index 1 for the actual word in this one-word sentence.
    # dataset_reader = predictor._dataset_reader
    # instance = dataset_reader.text_to_instance(sentence)
    # output = predictor.predict_instance(instance)
    # if hidden:
    #     return output["lm_embeddings"][1]  # Take index 1 for the actual word in this one-word sentence.
    # else:  # return logits


def featurize(sents1, sents2, predictor, hidden=False):
    features1 = np.array([_featurize(s, predictor, hidden) for s in sents1])
    features2 = np.array([_featurize(s, predictor, hidden) for s in sents2])
    return np.concatenate([features1, features2], axis=1)


def get_targets(sentence, predictor):
    tokens = get_instance(sentence, predictor)["source"].tokens
    token_ids = torch.Tensor([predictor._model.vocab.get_token_index(str(t)) for t in tokens])
    targets = torch.zeros_like(token_ids)
    targets[0:-1] = token_ids[1:]
    return targets


def score(sentence, predictor):
    lm_embeddings = get_lm_embeddings(sentence, predictor)
    h = torch.Tensor(lm_embeddings)
    # .chunk(2, -1)[0]    # Get only the forward hidden states
    targets = get_targets(sentence, predictor)
    probs = torch.nn.functional.log_softmax(
        torch.matmul(h, predictor._model._softmax_loss.softmax_w) + predictor._model._softmax_loss.softmax_b, dim=-1
    )
    return probs[range(len(probs)-2), targets.long()[:-2]]   # Throw away last two probabilities for stop and padding


def get_data(path):
    with open(path) as fh:
        tsv_contents = [line.strip().split("\t") for line in fh.readlines()]
    sents1, sents2, labels = zip(*tsv_contents)
    labels = np.array([1 if l == "True" else 0 for l in labels])
    return sents1, sents2, labels


def split(data):
    split_idx = len(data) // 2
    return data[:split_idx], data[split_idx:]

def test_logistic_regression(sents1, sents2, labels, n_splits=20):
    for _ in range(n_splits):
        sents1, sents2, labels = shuffle(sents1, sents2, labels)
        train_sents1, test_sents1 = split(sents1)
        train_sents2, test_sents2 = split(sents2)
        train_labels, test_labels = split(labels)
        for model_name in models:
            model_path = os.path.join(args.model_dir, model_name)
            predictor = get_predictor(model_path)
            train_features = featurize(train_sents1, train_sents2, predictor)
            test_features = featurize(test_sents1, test_sents2, predictor)
            clf = LogisticRegression().fit(train_features, train_labels)
            preds = clf.predict(test_features)
            acc = (preds == test_labels).mean()
            accs[model_name].append(acc)
        true_acc = (test_labels == 1).mean()
        false_acc = (test_labels == 0).mean()
        maj_acc = max(true_acc, false_acc)
        accs["baseline"].append(maj_acc)

    # plt.figure()
    for model, data in accs.items():
        print(f"{model} mean: {np.mean(data)}")

    sns.kdeplot(data=pd.DataFrame(accs))
    plt.show()

# def build_test_sentences(s1, s2, test):
#     if test == "uniform_true":  # [[x]] ⊆ [[y]] <==> p(xy) = p(xx)
#         test1 = f"{s1} {s2}"
#         test2 = f"{s1} {s1}"
#         return test1, test2
#     elif test == "independently_truthful":  # [[x]] ⊆ [[y]] <==> p(xy) / p(yT) = p(xx) / p(xT)
#         test1 =
#     elif test == "informative":  # [[x]] ⊆ [[y]] <==> p(y | x) / p(\epsilon | x) = p(y | y) / p(\epsilon | y)
#         pass

def scatterplot(lhs, rhs, labels, name):
    g = sns.scatterplot(x=lhs, y=rhs, hue=labels)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="center")
    g.grid(visible=True, which='major', color='black', linewidth=0.075)
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png")
    plt.clf()

def kdeplot(diff, labels, name):
    g = sns.kdeplot(x=diff, hue=labels)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="center")
    g.grid(visible=True, which='major', color='black', linewidth=0.075)
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png")
    plt.clf()

def auc(diff, labels):
    auc = Auc()
    auc(torch.Tensor(diff), torch.Tensor(labels))
    return auc.get_metric()


def test_entailment_uniform_true(sents1, sents2, labels):
    """[[x]] ⊆ [[y]] <==> p(xy) = p(xx)"""
    xy = [f"{x} {y}" for x, y in zip(sents1, sents2)]
    xx = [f"{x} {x}" for x in sents1]

    for model_name in models:
        model_path = os.path.join(args.model_dir, model_name)
        predictor = get_predictor(model_path)
        p_xy = [sum(score(s, predictor)).item() for s in xy]
        p_xx = [sum(score(s, predictor)).item() for s in xx]
        p_diff = [abs(a - b) for a, b in zip(p_xy, p_xx)]

        # g = sns.scatterplot(x=p_xy, y=p_xx, hue=labels)
        # g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="center")
        # g.grid(visible=True, which='major', color='black', linewidth=0.075)
        # plt.tight_layout()
        # plt.savefig(f"plots/{model_name}_uniform_true.png")
        # plt.clf()


def test_entailment_independent_truthful(sents1, sents2, labels):
    """[[x]] ⊆ [[y]] <==> p(xy) / p(yT) = p(xx) / p(xT)"""
    xy = [f"{x} {y}" for x, y in zip(sents1, sents2)]
    xx = [f"{x} {x}" for x in sents1]
    xT = sents1
    yT = sents2
    for model_name in models:
        model_path = os.path.join(args.model_dir, model_name)
        predictor = get_predictor(model_path)
        p_xy = [sum(score(s, predictor)).item() for s in xy]
        p_xx = [sum(score(s, predictor)).item() for s in xx]
    # TODO

def test_entailment_informative(sents1, sents2, labels):
    """[[x]] ⊆ [[y]] <==> p(y | x) / p(\\epsilon | x) = p(y | y) / p(\\epsilon | y)"""
    # TODO


# def test_uniform_true(sents1, sents2, labels):

if __name__ == "__main__":
    args = parse_args()
    models = [model_name for model_name in os.listdir(args.model_dir) if os.path.isdir(os.path.join(args.model_dir, model_name))]
    predictors = [get_predictor(os.path.join(args.model_dir, model)) for model in models]
    sents1, sents2, labels = get_data(args.eval_path)
    accs = defaultdict(list)

    sns.set_style()
    if args.analysis_method == "logistic_regression":
        test_logistic_regression(sents1, sents2, labels)
    elif args.analysis_method == "uniform_true":
        test_entailment_uniform_true(sents1, sents2, labels)

#     plt.hist(data, label=model, bins=list(range(min(data), max(data) + 1)), alpha=.2)
# plt.legend()
# plt.tight_layout()
# plt.show()

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
