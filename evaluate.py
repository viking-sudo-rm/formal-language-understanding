from copy import deepcopy
import os
import numpy as np
from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import product
import random
import torch

from allennlp.models.archival import load_archive
from allennlp.common.util import prepare_environment
from allennlp.predictors import Predictor
from allennlp_models.lm.dataset_readers import *

from src.quantifier.syntax import SimpleQuantifierSyntax
from src.quantifier.semantics import SimpleQuantifierSemantics, SimpleQuantifierWorld


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n_items", type=int, default=5, help="Number of entities.")
    parser.add_argument("--model_dir", type=str, default="models/quantifier/")
    parser.add_argument("--false_only", action="store_true")
    return parser.parse_args()


def contrast(sentence):
    """Modify the sentence by swapping """
    value = eval(sentence[-1])
    new_sent = deepcopy(sentence)
    new_sent[-1] = str(not value)
    return new_sent


def get_predictor(path, cuda=True):
    archive = load_archive(os.path.join(path, "model.tar.gz"), cuda_device=-1)
    config = deepcopy(archive.config)
    prepare_environment(config)
    model = archive.model
    model.eval()
    model.cuda()
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


def get_sentences(utterances):
    data = [
        (" ".join(u1), " ".join(u2), semantics.entails(tuple(u1), tuple(u2)))
        for u1, u2 in product(utterances, utterances)
    ]
    u1s, u2s, values = zip(*data)
    return list(u1s), list(u2s), np.array(list(values))


use_cuda = torch.cuda.is_available()
args = parse_args()
models = [
    model_name
    for model_name in os.listdir(args.model_dir)
    if os.path.isdir(os.path.join(args.model_dir, model_name))
]
predictors = [get_predictor(os.path.join(args.model_dir, model), cuda=use_cuda) for model in models]
syntax = SimpleQuantifierSyntax()
worlds = list(SimpleQuantifierWorld.generate_all(args.n_items))
semantics = SimpleQuantifierSemantics(worlds)
utterances = [u for u in syntax.generate() if u]

counts = defaultdict(int)
totals = defaultdict(int)
for _ in range(10):
    print("Random shuffle of utterances")
    random.shuffle(utterances)
    train_utterances = utterances[:3]
    test_utterances = utterances[3:]
    train_sents1, train_sents2, train_labels = get_sentences(train_utterances)
    test_sents1, test_sents2, test_labels = get_sentences(test_utterances)
    for model_name in models:
        model_path = os.path.join(args.model_dir, model_name)
        predictor = get_predictor(model_path)
        train_features = featurize(train_sents1, train_sents2, predictor)
        test_features = featurize(test_sents1, test_sents2, predictor)
        clf = LogisticRegression().fit(train_features, train_labels)
        preds = clf.predict(test_features)
        counts[model_name] += (preds == test_labels).sum()
        totals[model_name] += len(test_labels)
    counts["true"] = (test_labels == 1).sum()
    totals["true"] = len(test_labels)
    counts["false"] = (test_labels == 0).sum()
    totals["false"] += len(test_labels)

for model in counts:
    acc = counts[model] / totals[model]
    print(f"{model} mean: {acc}")
