"""Generate FOL language, then generate documents using RSA."""

from argparse import ArgumentParser
from tqdm import trange
import logging
import torch
from random import randint

from src.quantifier.syntax import QuantifierSyntax
from src.quantifier.semantics import QuantifierSemantics, QuantifierWorld
from src.rational_speech import RationalAgent, RationalSpeechActs
from src.only_true_speech import OnlyTrueAgent


logging.basicConfig()
log = logging.getLogger("fol_main")
log.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=1000)
    parser.add_argument("--doc_len", type=int, default=5)
    parser.add_argument("--n_items", type=int, default=5)
    parser.add_argument("--n_predicates", type=int, default=2)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--temp", type=float, default=1.)
    parser.add_argument("--cost", type=float, default=1/3)
    parser.add_argument("--no_updates", action="store_true")
    parser.add_argument("--only_true", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main(args):
    worlds = list(QuantifierWorld.generate_all(args.n_items, args.n_predicates))
    syntax = QuantifierSyntax(args.n_predicates)
    semantics = QuantifierSemantics(worlds)

    utterances = list(syntax.generate())
    truth_values = torch.tensor([[semantics.evaluate(e, w) for w in worlds] for e in utterances])
    costs = torch.tensor([args.cost * len(u) for u in utterances])
    belief_state = torch.zeros(len(worlds))

    rsa = RationalSpeechActs(utterances, truth_values, costs)
    agent = RationalAgent(rsa, temp=args.temp) if not args.only_true else OnlyTrueAgent(rsa)

    for _ in trange(args.n_docs):
        belief_state[randint(0, len(worlds) - 1)] = 1
        document = agent.sample_monologue(belief_state, length=args.doc_len, update_prior=not args.no_updates)
        sentences = [" ".join(str(x) for x in sent) for sent in document if sent]
        serialized = "; ".join(sent for sent in sentences)
        print(serialized)


if __name__ == "__main__":
    main(parse_args())
