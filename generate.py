"""Generate documents using the RSA speaker."""

from argparse import ArgumentParser
import tqdm
import logging
import torch
import random
from itertools import product

from src.quantifier.syntax import SimpleQuantifierSyntax
from src.quantifier.semantics import SimpleQuantifierSemantics, SimpleQuantifierWorld
from src.quantifier.serialize import to_string as q_to_string
from src.powerset.syntax import PowersetSyntax
from src.powerset.semantics import PowersetSemantics
from src.powerset.serialize import to_string as s_to_string
from src.binary.syntax import BinarySyntax
from src.binary.semantics import BinarySemantics
from src.binary.serialize import BinarySerializer
from src.rational_speech import RationalAgent, RationalSpeechActs


logging.basicConfig()
log = logging.getLogger("generate")
log.setLevel(logging.INFO)

languages = ["quantifier", "arithmetic", "powerset", "binary"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("lang", type=str, choices=languages, default="quantifier")
    parser.add_argument("-n", "--n_docs", type=int, default=100000, help="Number of total documents (training examples) to sample.")
    parser.add_argument("--n_items", type=int, default=5, help="Number of entities.")
    parser.add_argument("--cost", type=float, default=0, help="Cost per token.")
    parser.add_argument("--seed", type=int, default=2, help="Fixed random seed for data generation.")
    parser.add_argument("--temp", type=float, default=1., help="Temperature parameter in RSA model. Controls rationality.")
    parser.add_argument("--depth", type=int, default=1, help="Depth of RSA process, in the notation of the paper.")
    parser.add_argument("--vanilla", action="store_true", help="Whether to use vanilla RSA. Default if not set is to use vanilla anyway.")
    parser.add_argument("--noisy", action="store_true", help="Whether to use cRSA instead of normal RSA.")
    parser.add_argument("--dependent", action="store_true", help="Should second sentence depend on the first?")
    parser.add_argument("--eval", action="store_true", help="Create evaluation data.")
    return parser.parse_args()

class Generator:
    def __init__(self, seed, lang, n_items, cost, temp, depth, noisy, dependent):
        random.seed(seed)
        torch.manual_seed(seed)

        # self.seed = seed
        # self.lang = lang
        # self.n_items = n_items

        # Set up the model for the world and language's syntax and semantics.
        if lang == "quantifier":
            self.syntax = SimpleQuantifierSyntax()
            self.worlds = list(SimpleQuantifierWorld.generate_all(n_items))
            self.semantics = SimpleQuantifierSemantics(self.worlds)
            self.to_string = q_to_string
        elif lang == "powerset":
            self.syntax = PowersetSyntax(n_items)
            self.worlds = [w for w in range(n_items)]
            self.semantics = PowersetSemantics()
            self.to_string = s_to_string
        elif lang == "binary":
            self.syntax = BinarySyntax()
            self.worlds = [0, 1]
            self.semantics = BinarySemantics()
            self.to_string = BinarySerializer().to_string
        else:
            raise NotImplementedError(f"Unknown lang: {lang}.")
        self.utterances = list(self.syntax.generate())
        self.costs = torch.tensor([cost * self.syntax.get_cost(u) for u in self.utterances])
        self.truth_values = torch.tensor([[self.semantics.evaluate(e, w) for w in self.worlds] for e in self.utterances])
        self.errors = torch.tensor([.1 for _ in self.utterances])
        self.rsa = RationalSpeechActs(self.utterances, self.truth_values, self.costs, self.errors, self.syntax)
        self.agent = RationalAgent(self.rsa, temp=temp, depth=depth, noisy=noisy, conditional_independence=not dependent)


    def generate_one(self):
        world = random.randint(0, len(self.worlds) - 1)
        context = []
        for _ in range(512):
            curr_context = None if context == [] else context
            sent = self.agent.speak(world, context=curr_context)
            context.append(sent)
            if self.syntax.is_empty(sent):
                break
        serialized = " ".join(self.to_string(x) for x in context)  # if not self.syntax.is_empty(x))
        print(serialized)

    def generate_n(self, n_docs):
        for _ in tqdm.trange(n_docs):
            self.generate_one()

    def score(self, utterances, context=None):
        return self.agent.score_all(utterances, context)

@torch.no_grad()
def main(args):
    generator = Generator(args.seed, args.lang, args.n_items, args.cost, args.temp, args.depth, args.noisy, args.dependent)
    generator.generate_n(args.n_docs)




    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    #
    # # Set up the model for the world and language's syntax and semantics.
    # if args.lang == "quantifier":
    #     syntax = SimpleQuantifierSyntax()
    #     worlds = list(SimpleQuantifierWorld.generate_all(args.n_items))
    #     semantics = SimpleQuantifierSemantics(worlds)
    #     to_string = q_to_string
    # elif args.lang == "powerset":
    #     syntax = PowersetSyntax(args.n_items)
    #     worlds = [w for w in range(args.n_items)]
    #     semantics = PowersetSemantics()
    #     to_string = s_to_string
    # elif args.lang == "binary":
    #     syntax = BinarySyntax()
    #     worlds = [0, 1]
    #     semantics = BinarySemantics()
    #     to_string = BinarySerializer().to_string
    # else:
    #     raise NotImplementedError(f"Unknown lang: {args.lang}.")
    # utterances = list(syntax.generate())
    # costs = torch.tensor([args.cost * syntax.get_cost(u) for u in utterances])
    #
    # if args.eval:
    #     # Generate a semantic evaluation suite.
    #     pairs = [(u1, u2) for u1, u2 in product(utterances, utterances) if u1 and u2]
    #     for sent1, sent2 in tqdm.tqdm(pairs):
    #         value = semantics.entails(tuple(sent1), tuple(sent2))
    #         print(f"{to_string(sent1)}\t{to_string(sent2)}\t{value}")
    #     return
    #
    # truth_values = torch.tensor([[semantics.evaluate(e, w) for w in worlds] for e in utterances])
    # errors = torch.tensor([.1 for _ in utterances])
    # rsa = RationalSpeechActs(utterances, truth_values, costs, errors)
    # agent = RationalAgent(rsa, temp=args.temp, depth=args.depth, noisy=args.noisy, conditional_independence=not args.dependent)
    #
    # for _ in tqdm.trange(args.n_docs):
    #     world = random.randint(0, len(worlds) - 1)
    #     context = []
    #     for _ in range(512):
    #         curr_context = None if context == [] else context
    #         sent = agent.speak(world, context=curr_context)
    #         context.append(sent)
    #         if syntax.is_empty(sent):
    #             break
    #     serialized = " ".join(to_string(x) for x in context if not syntax.is_empty(x))
    #     print(serialized)


if __name__ == "__main__":
    main(parse_args())
