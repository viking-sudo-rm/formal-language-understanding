"""Generate FOL language, then generate documents using RSA."""

from argparse import ArgumentParser
import tqdm
import logging
import torch
import random
from itertools import product

from src.quantifier.syntax import SimpleQuantifierSyntax
from src.quantifier.semantics import SimpleQuantifierSemantics, SimpleQuantifierWorld
from src.quantifier.serialize import to_string
from src.rational_speech import RationalAgent, RationalSpeechActs
# from src.only_true_speech import OnlyTrueAgent


logging.basicConfig()
log = logging.getLogger("fol_main")
log.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument("agent", type=str, choices=["true", "rsa", "eval"], help="Agent model to use, or train/test for eval dataset.")
    parser.add_argument("--task", type=str, choices=["quantifier", "arithmetic"], default="quantifier")
    parser.add_argument("-n", "--n_docs", type=int, default=10000, help="Number of total documents (training examples) to sample.")
    parser.add_argument("--n_sents", type=int, default=10, help="Number of sentences per document.")
    parser.add_argument("--n_items", type=int, default=5, help="Number of entities.")
    # parser.add_argument("--n_predicates", type=int, default=2, help="Number of predicates.")
    parser.add_argument("--cost", type=float, default=0, help="Cost per token.")
    parser.add_argument("--seed", type=int, default=2, help="Fixed random seed for data generation.")
    parser.add_argument("--temp", type=float, default=1., help="Temperature parameter in RSA model. Controls rationality.")
    parser.add_argument("--depth", type=int, default=1, help="Depth of RSA process, in the notation of the paper.")
    parser.add_argument("--noisy", action="store_true", help="Whether to use cRSA instead of normal RSA.")
    parser.add_argument("--dependent", action="store_true", help="Should second sentence depend on the first?")
    parser.add_argument("--eval", action="store_true", help="Create evaluation data.")
    return parser.parse_args()


@torch.no_grad()
def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set up the model for the world and language's syntax and semantics.
    if args.task == "quantifier":
        syntax = SimpleQuantifierSyntax()
        worlds = list(SimpleQuantifierWorld.generate_all(args.n_items))
        semantics = SimpleQuantifierSemantics(worlds)
    elif args.task == "arithmetic":
        pass
    else:
        raise NotImplementedError(f"Unknown task: {args.task}.")
    utterances = list(syntax.generate())

    if args.eval:
        # Generate a semantic evaluation suite.
        pairs = [(u1, u2) for u1, u2 in product(utterances, utterances) if u1 and u2]
        for sent1, sent2 in tqdm.tqdm(pairs):
            value = semantics.entails(tuple(sent1), tuple(sent2))
            print(f"{to_string(sent1)}\t{to_string(sent2)}\t{value}")
        return

    truth_values = torch.tensor([[semantics.evaluate(e, w) for w in worlds] for e in utterances])
    costs = torch.tensor([args.cost * len(u) for u in utterances])
    errors = torch.tensor([.1 for _ in utterances])
    rsa = RationalSpeechActs(utterances, truth_values, costs, errors)
    agent = RationalAgent(rsa, temp=args.temp, depth=args.depth, noisy=args.noisy, conditional_independence=not args.dependent)
    for _ in tqdm.trange(args.n_docs):
        world = random.randint(0, len(worlds) - 1)
        sent1 = agent.speak(world)
        sent2 = agent.speak(world, context=[sent1])
        doc = sent1 + sent2
        serialized = " ".join(x for x in doc if x)
        print(serialized)


if __name__ == "__main__":
    main(parse_args())
