"""A grammar to generate sentences in a syntactically simple language of quantifiers."""

from typing import Iterator, List, Union
import random
from itertools import product


class PowersetSyntax:

    def __init__(self, n_worlds: int):
        self.n_worlds = n_worlds

    def generate(self) -> Iterator[List[int]]:
        for string in product(*[[0, 1] for _ in range(self.n_worlds)]):
            yield list(string)

    @staticmethod
    def get_cost(prop: List[int]):
        return prop.count(0)

    @staticmethod
    def is_empty(prop: List[int]):
        return all(w == 1 for w in prop)

    # def generate_entails(self):