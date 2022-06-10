"""Map sentence objects to/from string format."""

def to_string(prop):
    if type(prop) == list:
        if len(prop) == 0:
            return ""
        elif type(prop[0]) == list:
            return " ".join("".join(str(i) for i in word) for word in prop)
        else:
            return "".join(str(x) for x in prop)


def from_string(string):
    return [int(c) for c in string]

