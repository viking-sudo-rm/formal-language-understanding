from itertools import product
import pandas as pd
import argparse
import os

from src.powerset.semantics import PowersetSemantics
from src.powerset.syntax import PowersetSyntax
from src.powerset.serialize import to_string as p_to_string

from src.binary.semantics import BinarySemantics
from src.binary.syntax import BinarySyntax
from src.binary.serialize import BinarySerializer


parser = argparse.ArgumentParser()
parser.add_argument("lang", choices=["powerset", "binary"])
parser.add_argument("--n_items", type=int, default=5)
parser.add_argument("--data_dir", type=str, default="/scratch/wcm9940/synthetic-language-understanding/data")
args = parser.parse_args()

if args.lang == "powerset":
    semantics = PowersetSemantics()
    syntax = PowersetSyntax(args.n_items)
    to_string = p_to_string
elif args.lang == "binary":
    semantics = BinarySemantics()
    syntax = BinarySyntax()
    to_string = BinarySerializer().to_string
else:
    raise NotImplementedError
    

pairs = filter(lambda x: x[0] != x[1], product(syntax.generate(), syntax.generate()))
df = pd.DataFrame([(to_string(p[0]), to_string(p[1]), semantics.entails(p[0], p[1])) for p in pairs])

save_path = f"{args.data_dir}/{args.lang}-{args.n_items}/eval.tsv"
if not os.path.isdir(save_path):
    os.makedirs(save_path)
df.to_csv(save_path, index=False, header=False, sep="\t")
