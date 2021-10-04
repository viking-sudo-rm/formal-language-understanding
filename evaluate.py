from copy import deepcopy
import os
import numpy as np

from allennlp.models.archival import load_archive
from allennlp.common.util import prepare_environment
from allennlp.predictors import Predictor
from allennlp_models.lm.dataset_readers import *


model_path = "models/only_true/"
eval_path = "data/test.tsv"


def contrast(sentence):
    """Modify the sentence by swapping """
    value = eval(sentence[-1])
    new_sent = deepcopy(sentence)
    new_sent[-1] = str(not value)
    return new_sent


archive = load_archive(
    os.path.join(model_path, "model.tar.gz"),
    cuda_device=-1,
)
config = deepcopy(archive.config)
prepare_environment(config)
model = archive.model
model.eval()
dataset_reader = archive.validation_dataset_reader
predictor = Predictor(model, dataset_reader)

with open(eval_path) as fh:
    tsv_contents = [line.strip().split("\t") for line in fh.readlines()]
sents, labels = zip(*tsv_contents)
labels: list
instances = [dataset_reader.text_to_instance(sent) for sent in sents]

predictions = []
ground_truths = []
probabilities = []
while instances:
    instance1 = instances.pop(0)
    instance2 = instances.pop(0)
    label1 = labels.pop(0)
    label2 = labels.pop(0)
    loss1 = instance1["loss"]
    loss2 = instance2["loss"]
    predicted1 = loss1 < loss2
    cond_prob1 = np.exp2(loss1) / (np.exp2(loss1) + np.exp2(loss2))
    predictions.append(0 if predicted1 else 1)
    ground_truths.append(0 if label1 else 1)
    probabilities.append([cond_prob1, 1 - cond_prob1])

predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
probabilities = np.array(probabilities)

accuracy = np.sum(predictions == ground_truths) / len(predictions)
cross_ent = np.sum(-np.log(probabilities)[ground_truths]) / len(predictions)

print("Accuracy", accuracy)
print("Cross ent", cross_ent)
