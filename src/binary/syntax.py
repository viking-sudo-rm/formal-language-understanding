"""A grammar that generates binary sentences."""

from typing import Iterator, List


class BinarySyntax:

    def generate(self) -> Iterator[List[int]]:
        yield [0]
        yield [1]
        yield [0, 1]
        yield []

    @staticmethod
    def get_cost(prop: List[int]):
        return 0 if prop == [] else 1

    @staticmethod
    def is_empty(prop: List[int]):
        return prop == []

    # def generate_entails(self):