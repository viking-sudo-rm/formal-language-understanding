"""A grammar to generate sentences in a syntactically simple language of quantifiers."""

from typing import Iterator, List, Union
import random


class QuantifierSyntax:

    quantifiers = ["some", "most", "all", "none", "not_most", "not_all"]

    def __init__(self, n_predicates: int):
        self.n_predicates = n_predicates
    
    def generate(self) -> Iterator[List[Union[str, int]]]:
        yield []
        for quant in self.quantifiers:
            for p1 in range(self.n_predicates):
                for p2 in range(self.n_predicates):
                    yield [quant, p1, p2]
    
    def generate_entails(self):
        entail_sents = [["entails", s1, s2] for s1 in self.generate() for s2 in self.generate() if s1 and s2]
        random.shuffle(entail_sents)
        for sent in entail_sents:
            for val in [True, False]:
                yield sent + [val]


class SimpleQuantifierSyntax:

    quantifiers = ["some", "most", "all", "none", "not_most", "not_all"]
    
    def generate(self) -> Iterator[List[Union[str, int]]]:
        yield []
        for q in self.quantifiers:
            yield [q]
    
    def generate_entails(self):
        entail_sents = [["entails", s1, s2] for s1 in self.generate() for s2 in self.generate() if s1 and s2]
        random.shuffle(entail_sents)
        for sent in entail_sents:
            for val in [True, False]:
                yield sent + [val]

    @staticmethod
    def get_cost(prop):
        return len(prop)

    @staticmethod
    def is_empty(prop):
        return prop == []


class SimplePropositionSyntax:
    quantifiers = []

    def generate(self) -> Iterator[List[Union[str, int]]]:
        yield []
        for q in self.quantifiers:
            yield [q]

    def generate_entails(self):
        entail_sents = [["entails", s1, s2] for s1 in self.generate() for s2 in self.generate() if s1 and s2]
        random.shuffle(entail_sents)
        for sent in entail_sents:
            for val in [True, False]:
                yield sent + [val]
