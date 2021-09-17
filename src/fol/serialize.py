"""Convert expression trees to and from a readable tree format."""

QUANTIFIERS = {"all", "some"}
OPERATORS = {"and", "or", "not"}


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


def from_string(string, is_list=False):
    if not is_list:
        string = string.split(" ")
    
    stack = []
    while string:
        token = string.pop()
        if token == ")":
            full_const = stack.pop()
            stack[0].append(full_const)
        elif token in OPERATORS:
            string.pop()
            stack.push([token])
        elif token in QUANTIFIERS:
            var_name = string.pop()
            string.pop()
            stack.push([token, var_name])
        else:
            stack[0].append(token)
    
    assert len(stack) == 1
    return stack[0]
