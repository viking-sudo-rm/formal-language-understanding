"""Map sentence objects to/from string format."""


class BinarySerializer:

    @staticmethod
    def to_string(prop):
        return "".join(str(x) for x in prop)

    @staticmethod
    def from_string(string):
        return [int(c) for c in string]
