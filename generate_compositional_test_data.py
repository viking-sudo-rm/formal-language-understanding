import src.powerset.semantics as semantics
import src.powerset.syntax as syntax
from itertools import product, chain
import pandas as pd
from generate import Generator, languages
from src.powerset.serialize import to_string
from argparse import *
from math import log
from numpy import inf, prod
# import sys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("lang", type=str, choices=languages, default="quantifier")
    # parser.add_argument("-n", "--n_docs", type=int, default=100000, help="Number of total documents (training examples) to sample.")
    parser.add_argument("--n_items", type=int, default=5, help="Number of entities.")
    parser.add_argument("--cost", type=float, default=0, help="Cost per token.")
    parser.add_argument("--seed", type=int, default=2, help="Fixed random seed for data generation.")
    parser.add_argument("--temp", type=float, default=1., help="Temperature parameter in RSA model. Controls rationality.")
    parser.add_argument("--depth", type=int, default=1, help="Depth of RSA process, in the notation of the paper.")
    parser.add_argument("--vanilla", action="store_true", help="Whether to use vanilla RSA. Default if not set is to use vanilla anyway.")
    parser.add_argument("--noisy", action="store_true", help="Whether to use cRSA instead of normal RSA.")
    parser.add_argument("--dependent", action="store_true", help="Should second sentence depend on the first?")
    # parser.add_argument("--eval", action="store_true", help="Create evaluation data.")
    parser.add_argument("--eval_dir", type=str, help="Directory to write eval data in.")
    parser.add_argument("--max_sent_len", type=int, default=4, help="Maximum number of words to sample per utterance")
    # parser.add_argument("--downsample", type=int, default=1000, help="Sentences to use")
    return parser.parse_args()

args = parse_args()
generator = Generator(args.seed, args.lang, args.n_items, args.cost, args.temp, args.depth, args.noisy, args.dependent)
n_worlds = args.n_items
max_sent_len = args.max_sent_len
semantics = semantics.PowersetSemantics()
syntax = syntax.PowersetSyntax(n_worlds)
# to_string = lambda s: "".join(str(x) for x in s)
sentences = iter([])
vocab = [w for w in syntax.generate() if w != [1] * n_worlds]
for l in range(1, max_sent_len+1):
    sentences = chain(product(*(vocab for _ in range(l))), sentences)   # Generates a lot of contradictions which aren't interesting

def my_log(x):
    return -inf if x == 0 else log(x)

sentences = [list(s) + [[1] * n_worlds] for s in sentences if semantics.coordinate(s) != [0] * n_worlds]  # Filter out contradictions. This could be done more efficiently
probabilities = [my_log(generator.score(s)) for s in sentences]
df_probs = pd.DataFrame([(to_string(s), p) for s, p in zip(sentences, probabilities)], columns=("sentence", "logprob"))
df_probs.to_csv(f"{args.eval_dir}/eval_prob-{n_worlds}_worlds-{max_sent_len}_sents.tsv", index=False, sep="\t")
print(f"Utterances cover {prod(probabilities)} of probability mass")

pairs = filter(lambda x: x[0] != x[1], product(sentences, sentences))
df_pairs = pd.DataFrame([(to_string(p[0]), to_string(p[1]), semantics.entails_sent(p[0], p[1])) for p in pairs],
                        columns=("premise", "hypothesis", "entailment"))
df_pairs = df_pairs.merge(df_probs, left_on="premise", right_on="sentence")\
                   .merge(df_probs, left_on="hypothesis", right_on="sentence", suffixes=("_premise", "_hypothesis"))\
                   .drop(["sentence_premise", "sentence_hypothesis"], axis=1)

df_pairs.to_csv(f"{args.eval_dir}/eval_entail-{n_worlds}_worlds-{max_sent_len}_sents.tsv", index=False, sep="\t")
