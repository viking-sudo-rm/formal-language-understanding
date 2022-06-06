import semantics
import syntax
from itertools import product
import pandas as pd

semantics = semantics.PowersetSemantics()
syntax = syntax.PowersetSyntax(5)
to_string = lambda s: "".join(str(x) for x in s)

# pairs = filter(lambda x: x[0] != x[1], product(syntax.generate(), syntax.generate()))
# df = pd.DataFrame([(to_string(p[0]), to_string(p[1]), semantics.entails(p[0], p[1])) for p in pairs])
# df.to_csv("../../data/powerset/eval.tsv", index=False, header=False, sep="\t")