from typing import List, Iterator, Union, Set, Tuple
from itertools import product
from functools import lru_cache


Proposition = List[Union[str, int]]


class QuantifierWorld:
    def __init__(self, predicates: List[Set[int]], n_items: int):
        self.predicates = predicates
        self.n_items = n_items

    @property
    def n_predicates(self):
        return len(self.predicates)

    def measure(self, p1, p2) -> Tuple[int, int]:
        set1 = self.predicates[p1]
        set2 = self.predicates[p2]
        return len(set1 & set2), len(set1)

    @classmethod
    def generate_all(
        cls, n_items: int, n_predicates: int
    ) -> Iterator["QuantifierWorld"]:
        for assignment in product(
            *[[False, True] for _ in range(n_predicates * n_items)]
        ):
            predicates = []
            for pred_idx in range(n_predicates):
                start_idx = pred_idx * n_predicates
                items = {
                    idx
                    for idx, value in enumerate(
                        assignment[start_idx : start_idx + n_items]
                    )
                    if value
                }
                predicates.append(items)

            # Ignore weird edge cases with meaningless predicates.
            if all(len(pred) > 0 for pred in predicates):
                yield QuantifierWorld(predicates, n_items)


class QuantifierSemantics:

    quantifiers = {
        "some": lambda mu, tot: mu > 0,
        "most": lambda mu, tot: mu > tot / 2,
        "all": lambda mu, tot: mu == tot,
        "none": lambda mu, tot: mu == 0,
        "not_most": lambda mu, tot: mu < tot / 2,
        "not_all": lambda mu, tot: mu < tot,
    }

    def __init__(self, worlds):
        self.worlds = worlds

    def evaluate(self, prop: Proposition, world: QuantifierWorld) -> bool:
        if not prop:
            return True
        if prop[0] == "entails":
            _, q1, q2, val = prop
            return self.entails(tuple(q1), tuple(q2)) == val
        quant, p1, p2 = prop
        mu, tot = world.measure(p1, p2)
        quant_fn = self.quantifiers[quant]
        return quant_fn(mu, tot)

    @lru_cache(None)
    def entails(self, prop1: Proposition, prop2: Proposition) -> bool:
        """Here we use tuples instead of lists to allow memoization."""
        for world in self.worlds:
            if self.evaluate(prop1, world) and not self.evaluate(prop2, world):
                return False
        return True

    def get_mask_idx(self, expr: Proposition) -> int:
        assert expr[0] == "entails"
        return 2


class SimpleQuantifierSemantics:

    quantifiers = {
        "some": lambda mu, tot: mu > 0,
        "most": lambda mu, tot: mu > tot / 2,
        "all": lambda mu, tot: mu == tot,
        "none": lambda mu, tot: mu == 0,
        "not_most": lambda mu, tot: mu < tot / 2,
        "not_all": lambda mu, tot: mu < tot,
    }

    def __init__(self, worlds):
        self.worlds = worlds

    def evaluate(self, prop: Proposition, world: QuantifierWorld) -> bool:
        if not prop:
            return True
        mu, tot = world.measure(0, 0)
        quant, = prop
        quant_fn = self.quantifiers[quant]
        return quant_fn(mu, tot)

    @lru_cache(None)
    def entails(self, prop1: Proposition, prop2: Proposition) -> bool:
        """Here we use tuples instead of lists to allow memoization."""
        for world in self.worlds:
            if self.evaluate(prop1, world) and not self.evaluate(prop2, world):
                return False
        return True

    def get_mask_idx(self, expr: Proposition) -> int:
        assert expr[0] == "entails"
        return 2