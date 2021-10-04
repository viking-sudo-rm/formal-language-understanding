"""A grammar to generate sentences in a syntactically simple language of quantifiers."""

from typing import Iterator, List, Union
import random


class QuantifierSyntax:

    quantifiers = ["some", "most", "all", "none", "not_most", "not_all"]

    def __init__(self, n_predicates: int):
        self.n_predicates = n_predicates
    
    def generate(self, train: bool = True) -> Iterator[List[Union[str, int]]]:
        if train:
            yield []
            for quant in self.quantifiers:
                for p1 in range(self.n_predicates):
                    for p2 in range(self.n_predicates):
                        yield [quant, p1, p2]

        entail_sents = [["entails", q1, q2] for q1 in self.quantifiers for q2 in self.quantifiers]
        random.shuffle(entail_sents)
        for idx, sent in enumerate(entail_sents):
            for val in [True, False]:
                if train and idx % 2 == 0:
                    yield sent + [val]
                elif not train and idx % 2 == 1:
                    yield sent + [val]
