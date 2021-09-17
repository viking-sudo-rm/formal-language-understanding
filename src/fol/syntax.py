"""Implement a grammar to generate first-order logic (FOL) expressions."""

from itertools import product
from functools import lru_cache


def contains(expression, var):
    if isinstance(expression, int) and expression == var:
        return True
    if isinstance(expression, list):
        return any(contains(child, var) for child in expression)
    return False


def is_vacuous(expr):
    if isinstance(expr, list):
        if expr[0] == "some" or expr[0] == "all":
            return not contains(expr[2], expr[1])
        else:
            return any(is_vacuous(child) for child in expr)
    return False


class FolSyntax:

    def __init__(self, entities, predicates):
        self.predicates = predicates
        self.entities = entities

    @lru_cache(None)
    def populate_expressions(self, depth, curr_var_depth, var_depth):
        if depth == 0:
            vars = list(range(curr_var_depth))
            return [[pred, ent] for pred, ent in product(self.predicates, list(self.entities) + vars)]

        expressions = self.populate_expressions(depth - 1, curr_var_depth, var_depth)
        new_expressions = []
        for expr in expressions:
            new_expressions.append(["not", expr])
        for expr1, expr2 in product(expressions, expressions):
            new_expressions.append(["and", expr1, expr2])
            new_expressions.append(["or", expr1, expr2])
            
        if curr_var_depth < var_depth:
            bound_expressions = self.populate_expressions(depth - 1, curr_var_depth + 1, var_depth)
            for expr in bound_expressions:
                new_expressions.append(["all", curr_var_depth, expr])
                new_expressions.append(["some", curr_var_depth, expr])
        
        return new_expressions
    
    def generate(self, depth: int = 3, var_depth: int = 1, vacuous=False):
        yield None
        for d in range(depth + 1):
            for expr in self.populate_expressions(d, 0, var_depth):
                if vacuous or not is_vacuous(expr):
                   yield expr 
