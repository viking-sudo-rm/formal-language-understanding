"""Generate FOL language, then generate documents using RSA."""

from argparse import ArgumentParser
import logging
import torch
from random import randint

from src.fol.syntax import FolSyntax
from src.fol.semantics import FolSemantics, FolWorld
from src.fol.serialize import to_string
from src.fol.utils import size
from src.rational_speech import RationalAgent, RationalDialog


logging.basicConfig()
log = logging.getLogger("fol_main")
log.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--doc_len", type=int, default=5)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--var_depth", type=int, default=2)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--vacuous", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main(args):
    entities = ["john", "mary"]
    predicates = ["blue", "red"]
    syntax = FolSyntax(entities, predicates)
    semantics = FolSemantics(entities, predicates)

    worlds = list(FolWorld.generate_all(entities, predicates))  # Has length 2^n_params = 16.
    utterances = list(syntax.generate(depth=args.depth, var_depth=args.var_depth, vacuous=args.vacuous))
    truth_values = torch.tensor([[semantics.evaluate(e, w) for w in worlds] for e in utterances])
    # costs = torch.tensor([depth(u) + 1 for u in utterances])
    costs = torch.tensor([size(u) for u in utterances])
    belief_state = torch.zeros(len(worlds))
    belief_state[randint(0, len(worlds) - 1)] = 1
    world = worlds[belief_state.argmax()]

    dialog = RationalDialog(utterances, truth_values, costs)
    agent = RationalAgent(dialog)
    document = agent.sample_monologue(belief_state, length=args.doc_len)
    
    print("=" * 3, "World State", "=" * 3)
    print(world.pred_map)
    print()

    print("=" * 3, "Monologue", "=" * 3)
    for sentence in document:
        if args.eval:
            value = semantics.evaluate(sentence, world)
            print(to_string(sentence), "==", value)
        else:
            print(to_string(sentence))


if __name__ == "__main__":
    main(parse_args())
