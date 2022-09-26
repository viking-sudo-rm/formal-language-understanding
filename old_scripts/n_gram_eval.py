import numpy as np
from nltk.util import bigrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import pandas as pd
from math import log

order = 10
train_path = "data/powerset/dependent/train.txt"
text = [line.split() for line in open(train_path)]
train, vocab = padded_everygram_pipeline(order, text)
lm = MLE(order)
lm.fit(train, vocab)

def lm_score(sentence):
    """Score a sentence. Do not include padding, but may include model's own EOS token: 1^|w|"""
    p = 0
    context = ["<s>"] * (order - 1)
    for word in sentence.split():
        p += lm.logscore(word, context)
        context = context[1:] + [word]
    return p


eval_prob_path = "data/powerset/dependent/eval_prob-3_worlds-5_sents.tsv"
df_eval = pd.read_csv(eval_prob_path, sep="\t")
df_eval["logprob_lm"] = df_eval.sentence.apply(lm_score)

def error(row):
    if row.lm_prediction == 0 and row.probability == 0:
        return 0
    elif row.lm_prediction == 0 or row.probability == 0:
        return np.inf
    else:
        return abs(row.logprob - row.logprob_lm)

df_eval["logprob_error"] = df_eval.apply(error, axis=1)
pass