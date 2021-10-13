from unittest import TestCase

from src.quantifier.semantics import SimpleQuantifierWorld, SimpleQuantifierSemantics


class TestSimpleQuantifierSemantics(TestCase):

    def setUp(self):
        self.worlds = list(SimpleQuantifierWorld.generate_all(4))
        self.semantics = SimpleQuantifierSemantics(self.worlds)

    def test_some(self):
        values = [self.semantics.evaluate(("some",), world) for world in self.worlds]
        self.assertEqual(values, [False, True, True, True, True])

    def test_most(self):
        values = [self.semantics.evaluate(("most",), world) for world in self.worlds]
        self.assertEqual(values, [False, False, False, True, True])

    def test_all(self):
        values = [self.semantics.evaluate(("all",), world) for world in self.worlds]
        self.assertEqual(values, [False, False, False, False, True])

    def test_none(self):
        values = [self.semantics.evaluate(("none",), world) for world in self.worlds]
        self.assertEqual(values, [True, False, False, False, False])

    def test_not_most(self):
        values = [self.semantics.evaluate(("not_most",), world) for world in self.worlds]
        self.assertEqual(values, [True, True, False, False, False])

    def test_not_all(self):
        values = [self.semantics.evaluate(("not_all",), world) for world in self.worlds]
        self.assertEqual(values, [True, True, True, True, False])

    def test_entails(self):
        self.assertFalse(self.semantics.entails(("some",), ("most",)))
        self.assertTrue(self.semantics.entails(("none",), ("not_most",)))

