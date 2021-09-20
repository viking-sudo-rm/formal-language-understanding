"""Utilities defined over FOL syntactic trees."""


def size(tree: list) -> int:
    """The number of leaves in the tree."""
    if tree is None:
        return 0
    if isinstance(tree, str) or isinstance(tree, int):
        return 1
    return sum(size(child) for child in tree)


def depth(tree: list) -> int:
    """The longest rooted path in the tree."""
    if tree is None:
        return -1
    if isinstance(tree, str) or isinstance(tree, int):
        return 0
    return max(depth(child) + 1 for child in tree)
