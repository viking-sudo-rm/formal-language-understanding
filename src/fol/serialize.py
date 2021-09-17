"""Convert expression trees to and from a readable tree format."""

from typing import List, Union

QUANTIFIERS = {"all", "some"}


def to_string(tree, output_list=False):
    if not output_list:
        return " ".join(to_string(tree, output_list=True))
    if isinstance(tree, str) or tree is None:
        return [str(tree)]
    if isinstance(tree, int):
        return [f"x{tree}"]

    split = 2 if tree[0] == "all" or tree[0] == "some" else 1
    output = [f"x{x}" if isinstance(x, int) else x for x in tree[:split]]
    output.append("(")
    for child in tree[split:]:
        output.extend(to_string(child, output_list=True))
    output.append(")")
    return output


def from_string(string: Union[str, List[str]], is_list=False):
    if not is_list:
        string = string.split(" ")
    
    stack: List[list] = []
    while string:
        token = string.pop(0)
        if token == ")":
            full_const = stack.pop(0)
            if stack:
                stack[0].append(full_const)
            else:
                assert not string
                return full_const
        elif token in QUANTIFIERS:
            var_name = string.pop(0)
            string.pop(0)
            stack.append([token, var_name])
        elif string and string[0] == "(":
            string.pop(0)
            stack.append([token])
        else:
            stack[0].append(token)
