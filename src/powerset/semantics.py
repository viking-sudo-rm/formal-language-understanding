from typing import List, Iterator, Union, Set, Tuple, NamedTuple
from itertools import product
from functools import lru_cache


Proposition = List[int]


class PowersetSemantics:

    @staticmethod
    def evaluate(prop: Proposition, world: int) -> bool:
        return bool(prop[world])

    @staticmethod
    def entails(prop1: Proposition, prop2: Proposition) -> bool:
        """Here we use tuples instead of lists to allow memoization."""
        return all(w1 <= w2 for w1, w2 in zip(prop1, prop2))
