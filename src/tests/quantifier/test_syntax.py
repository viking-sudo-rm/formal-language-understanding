from src.quantifier.syntax import QuantifierSyntax
from unittest import TestCase
import random

from src.quantifier.syntax import QuantifierSyntax


class TestQuantifierSyntax(TestCase):

    def test_1_predicate(self):
        syntax = QuantifierSyntax(1)
        random.seed(2)
        train = list(syntax.generate(train=True))
        random.seed(2)
        test = list(syntax.generate(train=False))
        train_entails = {" ".join(s) for s in train if s and s[0] == "entails"}
        test_entails = {" ".join(s) for s in test}
        self.assertEqual(train_entails & test_entails, set())
