import semantics
import syntax
from itertools import product
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_items", type=int, default=5)
parser.add_argument("--save_path", type=str, default="/scratch/wcm9940/synthetic-language-understanding/data/powerset/eval.tsv")
args = parser.parse_args()

semantics = semantics.PowersetSemantics()
syntax = syntax.PowersetSyntax(args.n_items)
to_string = lambda s: "".join(str(x) for x in s)

pairs = filter(lambda x: x[0] != x[1], product(syntax.generate(), syntax.generate()))
df = pd.DataFrame([(to_string(p[0]), to_string(p[1]), semantics.entails(p[0], p[1])) for p in pairs])
df.to_csv(args.save_path, index=False, header=False, sep="\t")
