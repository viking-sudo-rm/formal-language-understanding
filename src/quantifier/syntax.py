"""A grammar to generate sentences in a syntactically simple language of quantifiers."""

from typing import Iterator, List, Union


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
