"""Generate predictions."""

from math import log, exp
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


plt.rcParams["font.family"] = "Times New Roman"


class SampleEstimator:

    def __init__(self,
                 eps: float = .1,
                 delta: float = .1,
                 ppl: float = 20.,
                 p_true: float = .5,
                ):
        self.eps = eps  # Error tolerance of bound.
        self.delta = delta  # Probability for bound to hold.
        self.ppl = ppl  # Perplexity of an average word. 20 is estimated by GPT-3.
        self.p_true = p_true  # Probability that a sentence is true. .5 is conservative.
        self.vocab_size = ppl  # Size of vocabulary in English.

    def get_n_ppl(self, l):
        """Estimate number of samples needed to estimate probability using PPL."""
        p = self.get_prob(l)
        n_eps = 64 * 2 * 1/p * 1/self.eps * 1/self.eps * 2/self.delta
        n_delta = log(self.delta / 2, 1 - p)
        return max(n_eps, n_delta)

    def get_prob(self, length):
        """Rough estimate of sentence length given probability, assuming 20 perplexity."""
        return (1/self.ppl)**(length + 1)

    def get_n_thm4(self, l):
        """Estimate number of samples needed to estimate probability using Gricean speaker rules."""
        # cost_coeff = log(self.ppl + 1)
        n_eps = 64 * 2 * (self.ppl + 1)**(l + 1) * 1/self.p_true * 1/self.eps * 1/self.eps * 2/self.delta
        # Numerical precision issues here, but not really necessary to estimate.
        # p = 1 / (exp(self.w * (l + 1)) + 1e-9)
        # n_delta = log(self.delta / 2, 1 - p)
        # return max(n_eps, n_delta)
        return n_eps


def main(args):
    est = SampleEstimator()

    lengths = list(range(1, 11))
    lengths_ = np.linspace(1, 11, 300)

    # Perplexity-based bound.
    ppl_ns = np.array([est.get_n_ppl(l) for l in lengths])
    ppl_ns_ = np.array([est.get_n_ppl(l) for l in lengths_])

    # Using Theorem 4 bound.
    ns = np.array([est.get_n_thm4(l) for l in lengths])
    ns_ = np.array([est.get_n_thm4(l) for l in lengths_])

    for idx, (n, ppl_n) in enumerate(zip(ns, ppl_ns)):
        print(idx + 1, ": log n =", log(n, 10), "log ppl_n =", log(ppl_n, 10))

    gpt3 = 10000000000
    plt.figure()
    plt.title(fR"Entailment sample complexity (error $\leq{est.eps}$, prob. $\geq{1 - est.delta}$)", fontsize=18)
    plt.fill_between(lengths_, ns_, gpt3, where=ns_<=gpt3, color="blue", alpha=.2)
    plt.plot(lengths, ns, color="blue", marker=".", label="Theorem 3 bound")
    # plt.fill_between(lengths_, ppl_ns_, gpt3, where=ppl_ns_<=gpt3, color="red", alpha=.2)
    # plt.plot(lengths, ppl_ns, color="red", marker=".", label="GPT-3 PPL bound")
    plt.axhline(y=gpt3, linestyle='dashed', label="GPT-3 training data", color="black")
    plt.xlabel("Sentence length", fontsize=16)
    plt.xticks(lengths)
    plt.ylabel("Training sentences required", fontsize=16)
    plt.yscale("log")
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--ppl", action="store_true")
    args = parser.parse_args()
    main(args)