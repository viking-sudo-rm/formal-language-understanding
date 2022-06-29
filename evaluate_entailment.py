import os

from generate import Generator, languages
from argparse import ArgumentParser
import pandas as pd
from src.evaluation.distributional_model import *
from src.powerset.serialize import from_string
from allennlp.training.metrics.auc import Auc
import torch
from numpy.random import choice
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test_data", type=str, help=".tsv file containing labeled entailment pairs")
    parser.add_argument("--distributional_model", type=str, help="Options: RSA (TODO: ngram, text_frequency, neural_lm)")
    parser.add_argument("--save_dir", type=str)

    # Params for what kind of analysis to do
    parser.add_argument("--incremental_eval", action="store_true", help="Should train LMs incrementally and evaluate individual sentences")
    parser.add_argument("--auc", action="store_true", help="Should compute AUC for each model")
    parser.add_argument("--plot_type", nargs="+", type=str, help="Options: line, scatter")
    parser.add_argument("--downsample", type=int, help="how many test pairs to use")
    parser.add_argument("--complexity", type=str, help="What complexity measure to compute & use. Options: length, surprisal, surprisal_bin")
    parser.add_argument("--n_increments", type=int, help="how many incremental models to train")



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
    parser.add_argument("--order", type=int, help="Order of ngram model, i.e. value of n")
    parser.add_argument("--training_path", type=str, help="Path to training data")
    parser.add_argument("--training_dir", type=str, help="Path to training data directory for more than one training set")
    parser.add_argument("--size", type=int, help="Size of training set. The name of the training set should be <TRAINING_PATH>/<SIZE>.txt")

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

    # model.test_gricean("001", "011 011")
    # model.score("001")
    print("Reading test data")
    test_data = pd.read_csv(args.test_data, sep="\t")
    if args.downsample:
        test_data = test_data.sample(args.downsample)
    # if args.incremental_eval:
    EPSILON = 0.1
    models = {}
    if args.distributional_model == "ngram":
        size = args.size
        train_path = os.path.join(args.training_dir, str(size)) + ".txt"
        print("Reading training data")
        text = [line.split() for line in open(train_path)]
        print("Reading training data complete")
        for i in range(args.n_increments):
            print(f"START:Training ngram model {i+1}/{args.n_increments}")
            ngram = NgramModel.train_lm(args.order, text=text)
            model = NgramModel(empty="1" * args.n_items, lm=ngram)
            models[size] = model
            size //= 2
            text = choice(text, replace=False, size=size)
            print(f"END:Training ngram model {i+1}/{args.n_increments}")
    elif args.distributional_model == "neural_lm":
        pass  # TODO
    elif args.distributional_model == "text_frequency":
        empty = "1" * args.n_items
        size = args.size
        train_path = os.path.join(args.training_dir, str(size)) + ".txt"
        text = [line.strip().removesuffix(empty).strip() for line in open(train_path)]
        for i in range(args.n_increments):
            print(f"START:Training text frequency model {i+1}/{args.n_increments}")
            model = TextFrequency(empty="1" * args.n_items, text=text)
            models[size] = model
            size //= 2
            text = choice(text, replace=False, size=size)
            print(f"END:Training text frequency model {i+1}/{args.n_increments}")
    # if args.auc:
    #     for model in models.keys():
    #         pred = test_data.apply(lambda x: models[model].test_gricean(x.premise, x.hypothesis), axis=1)
    #         auc_score = auc(list(pred), [int(not x) for x in test_data.entailment])
    #         print(auc_score)

    preds = {}
    for model in models.keys():
        preds[model] = test_data.apply(lambda x: models[model].test_gricean(x.premise, x.hypothesis, log_ratio=True), axis=1)
        if args.auc:
            auc_score = auc(list(preds[model]), [int(not x) for x in test_data.entailment])
            print(f"{model}: ", auc_score)


    df = pd.concat([pd.DataFrame(preds), test_data], axis=1)

    # Ugly scatterplot
    # sns.relplot(data=df, x="level_4", y=0, hue="length", col="entailment")
    # plt.show()

    if args.complexity == "length":
        df["complexity"] = df.apply(lambda x: len(x.premise.split()) + len(x.hypothesis.split()), axis=1)
        COMPLEXITY = "Sentence\nlength"
    elif args.complexity == "surprisal_bin":
        df["complexity"] = pd.qcut(df.apply(lambda x: -1*(x.logprob_premise + x.logprob_hypothesis), axis=1),
                                   q=7,
                                   labels=False)
        COMPLEXITY = "Complexity (Relative rank by surprisal)"
    elif args.complexity == "surprisal":
        df["complexity"] = df.apply(lambda x: -1*(x.logprob_premise + x.logprob_hypothesis), axis=1)
        COMPLEXITY = "Complexity (surprisal)"


    if "line" in args.plot_type:
        df = df.set_index(list(df.columns[-6:])).stack().reset_index().rename({"level_6": "n", 0: "distance"}, axis=1)
        plt.rcParams["font.family"] = "Times New Roman"
        g = sns.relplot(data=df[~(df.distance == 0)], x="n", y="distance", hue="complexity", kind="line", col="entailment", palette="crest",
                        height=2.5, aspect=1,)
        g.set(xscale="log", yscale="log")
        g._legend.set_title(COMPLEXITY)
        for ax in g.axes.flat:
            ax.set_xlabel("Training sentences")

        g.axes[0, 0].set_title(r"$x \nsubseteq y$")
        g.axes[0, 1].set_title(r"$x \subseteq y$")

        g.axes[0, 0].set_ylabel(r"$g_\hat{p}(xy)$")
        plt.savefig(f"plots/line_{args.distributional_model}_{args.size}.pdf")

        for ax in g.axes.flat:
            ax.set_ylim(0.01, 100)
        plt.savefig(f"plots/line_{args.distributional_model}_{args.size}_scale.pdf")
        plt.clf()

    if "min_n" in args.plot_type:
        # df.set_index(list(df.columns[-6:])).apply(lambda x: min([k for k in dict(x) if dict(x)[k] < 0.1 and dict(x)[k]>=0]), axis=1)
        # z = df.set_index(list(df.columns[-6:])).apply(lambda x: dict(x), axis=1).iloc[2]
        def f(x):
            d = dict(x)
            sizes = [k for k in d if d[k] <= 1 and d[k] > 0]
            if len(sizes) > 0:
                return min(sizes)
            else:
                return None
        df = df.set_index(list(df.columns[-6:])).apply(f, axis=1).reset_index().rename({0: "min_n"}, axis=1)
        from numpy import isnan
        df = df[df.min_n.apply(lambda x: not isnan(x))]
        g = sns.relplot(data=df, x="complexity", y="min_n", kind="scatter", col="entailment")
        g.set(yscale="log")
        plt.savefig(f"plots/min_n{args.distributional_model}_{args.size}.pdf")
        plt.clf()




    # if "line" in args.plot_type:
    #     df = df.set_index(list(df.columns[-4:])).stack().reset_index().rename({"level_4": "n", 0: "distance"}, axis=1)
    #     g = sns.relplot(data=df[~(df.distance == 0)], x="n", y="distance", hue="length", kind="line", col="entailment", palette="crest",)
    #                     # units="premise", estimator=None)
    #     g.set(xscale="log", yscale="log", ylim=(1e-3, 1e3))
    #     plt.savefig("../../plots/line.pdf")
    #     plt.clf()



        x = 1
        pass

    pass
