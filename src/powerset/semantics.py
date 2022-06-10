from typing import List, Iterator, Union, Set, Tuple, NamedTuple
from itertools import product
from functools import lru_cache, reduce


Proposition = List[int]


class PowersetSemantics:

    @staticmethod
    def evaluate(prop: Proposition, world: int) -> bool:
        return bool(prop[world])

    @staticmethod
    def entails(prop1: Proposition, prop2: Proposition) -> bool:
        return all(w1 <= w2 for w1, w2 in zip(prop1, prop2))

    @staticmethod
    def coordinate(sent: list[Proposition]) -> Proposition:
        return reduce(lambda p, q: [w1 and w2 for w1, w2 in zip(p, q)], sent)

    @staticmethod
    def entails_sent(prop1: list[Proposition], prop2: list[Proposition]) -> bool:
        return PowersetSemantics.entails(PowersetSemantics.coordinate(prop1), PowersetSemantics.coordinate(prop2))
