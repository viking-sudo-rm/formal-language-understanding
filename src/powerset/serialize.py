"""Map sentence objects to/from string format."""

def to_string(prop):
    return "".join(str(x) for x in prop)


def from_string(string):
    return [int(c) for c in string]
