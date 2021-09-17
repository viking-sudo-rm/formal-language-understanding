from unittest import TestCase

from src.fol.semantics import FolSemantics, FolWorld


class TestSemantics(TestCase):

    entities = ["john", "mary"]
    predicates = ["blue", "red"]
    pred_map = {
        ("blue", "john"): True,
        ("blue", "mary"): False,
        ("red", "john"): False,
        ("red", "mary"): True,
    }
    world = FolWorld(entities, predicates, pred_map, {})
    semantics = FolSemantics(entities, predicates)

    def test_predicates(self):
        self.assertEqual(self.semantics.evaluate(["blue", "john"], self.world), True)
        self.assertEqual(self.semantics.evaluate(["blue", "mary"], self.world), False)
        self.assertEqual(self.semantics.evaluate(["red", "john"], self.world), False)
        self.assertEqual(self.semantics.evaluate(["red", "mary"], self.world), True)

    def test_and(self):
        true_expr = ["and", ["blue", "john"], ["red", "mary"]]
        self.assertEqual(self.semantics.evaluate(true_expr, self.world), True)
        false_expr = ["and", ["blue", "john"], ["blue", "mary"]]
        self.assertEqual(self.semantics.evaluate(false_expr, self.world), False)

    def test_or(self):
        true_expr = ["or", ["blue", "john"], ["blue", "mary"]]
        self.assertEqual(self.semantics.evaluate(true_expr, self.world), True)
        false_expr = ["or", ["blue", "mary"], ["red", "john"]]
        self.assertEqual(self.semantics.evaluate(false_expr, self.world), False)

    def test_quantifiers(self):
        self.assertEqual(self.semantics.evaluate(["some", 0, ["blue", 0]], self.world), True)
        self.assertEqual(self.semantics.evaluate(["all", 0, ["blue", 0]], self.world), False)
