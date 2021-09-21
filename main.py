"""Generate FOL language, then generate documents using RSA."""

from argparse import ArgumentParser
import logging
import torch
from random import randint

from src.quantifier.syntax import QuantifierSyntax
from src.quantifier.semantics import QuantifierSemantics, QuantifierWorld
from src.rational_speech import RationalAgent, RationalSpeechActs


logging.basicConfig()
log = logging.getLogger("fol_main")
log.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--doc_len", type=int, default=5)
    parser.add_argument("--n_items", type=int, default=5)
    parser.add_argument("--n_predicates", type=int, default=2)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--temp", type=float, default=1.)
    parser.add_argument("--cost", type=float, default=1/3)
    parser.add_argument("--no_updates", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main(args):
    syntax = QuantifierSyntax(args.n_predicates)
    semantics = QuantifierSemantics()

    worlds = list(QuantifierWorld.generate_all(args.n_items, args.n_predicates))
    utterances = list(syntax.generate())
    truth_values = torch.tensor([[semantics.evaluate(e, w) for w in worlds] for e in utterances])
    costs = torch.tensor([args.cost * len(u) for u in utterances])
    belief_state = torch.zeros(len(worlds))
    belief_state[randint(0, len(worlds) - 1)] = 1
    world = worlds[belief_state.argmax()]

    rsa = RationalSpeechActs(utterances, truth_values, costs)
    agent = RationalAgent(rsa, temp=args.temp)
    document = agent.sample_monologue(belief_state, length=args.doc_len, update_prior=not args.no_updates)
    
    print("=" * 3, "World State", "=" * 3)
    print({idx: pred for idx, pred in enumerate(world.predicates)})
    print()

    print("=" * 3, "Monologue", "=" * 3)
    for sentence in document:
        if args.eval:
            value = semantics.evaluate(sentence, world)
            print(repr(" ".join(str(x) for x in sentence)), "==", value)
        else:
            print(repr(" ".join(str(x) for x in sentence)))

if __name__ == "__main__":
    main(parse_args())
