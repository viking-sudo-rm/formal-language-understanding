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
        quant, p1, p2 = prop
        if quant == "entails":
            return self.entails(p1, p2)
        mu, tot = world.measure(p1, p2)
        quant_fn = self.quantifiers[quant]
        return quant_fn(mu, tot)

    @lru_cache(None)
    def entails(self, q1, q2) -> bool:
        for world in self.worlds:
            for p1 in range(world.n_predicates):
                for p2 in range(world.n_predicates):
                    mu, tot = world.measure(p1, p2)
                    if self.quantifiers[q1](mu, tot) and not self.quantifiers[q2](mu, tot):
                        return False
        return True
