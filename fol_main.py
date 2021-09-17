"""Generate FOL language, then generate documents using RSA."""

from argparse import ArgumentParser
import logging
import torch
from random import randint

from src.fol.syntax import FolSyntax
from src.fol.semantics import FolSemantics, FolWorld
from src.fol.serialize import to_string
from src.rational_speech import sample_document


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


def depth(tree):
    if tree is None:
        return -1
    if isinstance(tree, str) or isinstance(tree, int):
        return 0
    return max(depth(child) for child in tree)


@torch.no_grad()
def main(args):
    entities = ["john", "mary"]
    predicates = ["blue", "red"]
    syntax = FolSyntax(entities, predicates)
    semantics = FolSemantics(entities, predicates)

    worlds = list(FolWorld.generate_all(entities, predicates))  # Has length 2^n_params = 16.
    utterances = list(syntax.generate(depth=args.depth, var_depth=args.var_depth, vacuous=args.vacuous))
    truth_values = torch.tensor([[semantics.evaluate(e, w) for w in worlds] for e in utterances])
    costs = torch.tensor([depth(u) + 1 for u in utterances])
    belief_state = torch.zeros(len(worlds))
    belief_state[randint(0, len(worlds) - 1)] = 1
    document = sample_document(utterances, truth_values, costs, belief_state, length=args.doc_len)
    
    print(worlds[belief_state.argmax()].pred_map)
    for sentence in document:
        print(to_string(sentence))


if __name__ == "__main__":
    main(parse_args())
