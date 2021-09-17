"""Classes for evaluating first-order logic (FOL) expressions."""

import operator
from typing import Any, List
from copy import copy
from itertools import product


class FolWorld:
    def __init__(self, entities, predicates, pred_map, var_map):
        self.entities = entities
        self.predicates = predicates
        self.pred_map = pred_map
        self.var_map = var_map
    
    def evaluate(self, pred: str, ent: str) -> Any:
        return self.pred_map[pred, ent]
    
    @classmethod
    def generate_all(cls, entities, predicates) -> List["FolWorld"]:
        n_params = len(entities) * len(predicates)
        for assignment in product(*([[0, 1]] * n_params)):
            pred_map = {}
            for i, pred in enumerate(predicates):
                for j, ent in enumerate(entities):
                    idx = i * len(entities) + j
                    pred_map[pred, ent] = assignment[idx]
            yield cls(entities, predicates, pred_map, {})


class Predicate:
    """Need this wrapper to avoid bound lambda issue."""

    def __init__(self, name):
        self.name = name
    
    def __call__(self, world):
        return lambda entity: world.evaluate(self.name, entity)


class FolSemantics:

    DENOTATIONS = {
        "not": lambda _: operator.not_,
        "and": lambda _: operator.and_,
        "or": lambda _: operator.or_,
    }

    def __init__(self, entities, predicates):
        self.predicates = predicates
        self.entities = entities
        self.denotations = copy(self.DENOTATIONS)
        for ent in self.entities:
            self.denotations[ent] = lambda _: ent
        for pred in self.predicates:
            self.denotations[pred] = Predicate(pred)
    
    def evaluate(self, tree, world):
        if tree is None:
            return True
        if isinstance(tree, str):
            return self.denotations[tree](world)
        if isinstance(tree, int):
            return world.var_map[tree]
        if tree[0] == "all" or tree[0] == "some":
            return self._evaluate_quantifier(*tree, world)

        values = [self.evaluate(child, world) for child in tree]
        return values[0](*values[1:])
    
    def _evaluate_quantifier(self, quant, var, tree, world):
        for ent in world.entities:
            world.var_map[var] = ent
            truth_value = self.evaluate(tree, world)
            del world.var_map[var]
            if quant == "all" and not truth_value:
                return False
            if quant == "some" and truth_value:
                return True
        return quant == "all"
