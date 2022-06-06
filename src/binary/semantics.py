from typing import List, Iterator, Union, Set, Tuple, NamedTuple
from itertools import product
from functools import lru_cache


Proposition = List[int]


class BinarySemantics:

    def evaluate(self, prop: Proposition, world: int) -> bool:
        return prop == [] or world in prop

    def entails(self, prop1: Proposition, prop2: Proposition) -> bool:
        return all(not self.evaluate(prop1, w) or self.evaluate(prop2, w) for w in [0, 1])
