"""Map sentence objects to/from string format."""
import re

def to_string(prop):
    if type(prop) == list:
        if len(prop) == 0:
            return ""
        elif type(prop[0]) == list:
            return " ".join("".join(str(i) for i in word) for word in prop)
        else:
            return "".join(str(x) for x in prop)


def from_string(string):
    # if re.fullmatch(r"\d+", string):    # it's just one "word"
    #     return [int(c) for c in string]
    # el
    if re.fullmatch(r"((\d+)\s?)+", string):  # it's multiple space-separated words
        return [[int(c) for c in word] for word in string.split()]
    else:
        raise ValueError("The input string is not correctly formatted. "
                         "It should be either a string of 0s and 1s, "
                         "or a string of space separated strings of 0s and 1s.")
