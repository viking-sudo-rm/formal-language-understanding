from distributional_model import *
from generate import Generator, languages
from argparse import ArgumentParser
import pandas as pd
from distributional_model import *
from src.powerset.serialize import from_string
from allennlp.training.metrics.auc import Auc
import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test_data", type=str, help=".tsv file containing labeled entailment pairs")
    parser.add_argument("--distributional_model", type=str, help="Options: RSA (TODO: ngram, text_frequency, neural_lm)")

    # Params for RSA distributional model
    parser.add_argument("--lang", type=str, choices=languages, default="quantifier")
    parser.add_argument("--n_items", type=int, default=5, help="Number of entities.")
    parser.add_argument("--cost", type=float, default=0, help="Cost per token.")
    parser.add_argument("--seed", type=int, default=2, help="Fixed random seed for data generation.")
    parser.add_argument("--temp", type=float, default=1., help="Temperature parameter in RSA model. Controls rationality.")
    parser.add_argument("--depth", type=int, default=1, help="Depth of RSA process, in the notation of the paper.")
    parser.add_argument("--vanilla", action="store_true", help="Whether to use vanilla RSA. Default if not set is to use vanilla anyway.")
    parser.add_argument("--noisy", action="store_true", help="Whether to use cRSA instead of normal RSA.")
    parser.add_argument("--dependent", action="store_true", help="Should second sentence depend on the first?")

    # Params for ngram distributional model
    # TODO

    # Params for text frequency distributional model
    # TODO

    # Params for neural LM distributional model
    # TODO


    return parser.parse_args()


def auc(diff, labels):
    auc = Auc()
    auc(torch.Tensor(diff), torch.Tensor(labels))
    return auc.get_metric()

if __name__ == "__main__":
    args = parse_args()
    if args.distributional_model == "RSA":
        rsa = Generator(args.seed, args.lang, args.n_items, args.cost, args.temp, args.depth, args.noisy, args.dependent)
        model = RSAModel(empty="1" * args.n_items, rational_agent=rsa)
    elif args.distributional_model == "ngram":
        pass  #TODO
    elif args.distributional_model == "text_frequency":
        pass  #TODO
    elif args.distributional_model == "neural_lm":
        pass  #TODO

    # model.test_gricean("001", "011 011")
    # model.score("001")
    test_data = pd.read_csv(args.test_data, sep="\t")
    test_data = test_data.sample(100000)
    test_data["pred"] = test_data.apply(lambda x: model.test_gricean(x.premise, x.hypothesis), axis=1)
    auc = auc(test_data.pred.to_list, [int(not x) for x in test_data.entailment])
    print(auc)
    pass