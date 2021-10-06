"Utilities for serializing propositions."

def to_string(prop) -> str:
    if isinstance(prop, list):
        return " ".join(to_string(x) for x in prop)
    return str(prop)
