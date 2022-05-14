"""Check the quality of a 2-gram language model as a function of the number of data."""

from collections import defaultdict
import numpy as np
from math import log, exp
import matplotlib.pyplot as plt


speaker = "informative"
speaker = "literal"
path = f"data/{speaker}-10000.txt"
with open(path) as fh:
    lines = [line.strip().split() for line in fh.readlines()]


NEG_INF = -float("inf")
PAIRS = {
    # ("0", "1"): False,
    # ("1", "0"): False,
    ("01", "1"): False,
    ("01", "0"): False,
    ("0", "01"): True,
    ("1", "01"): True,
}


def my_log(x):
    return NEG_INF if x == 0 else log(x)


def get_prob(key, counts):
    return counts[key] / counts[key[:-1]]


def test_literal(x, y, counts):
    pxy = get_prob((x, y), counts)
    pxE = get_prob((x, "E"), counts)
    return abs(my_log(pxy) - my_log(pxE))


def test_informative(x, y, counts):
    pxy = get_prob((x, y), counts)
    pxE = get_prob((x, "E"), counts)
    pyy = get_prob((y, y), counts)
    pyE = get_prob((y, "E"), counts)
    left = my_log(pxy) - my_log(pxE)
    right = my_log(pyy) - my_log(pyE)
    return abs(left - right)

def get_loss(counts):
    lit_entails, lit_not_entails = [], []
    inf_entails, inf_not_entails = [], []
    for (x, y), label in PAIRS.items():
        lit_diff = test_literal(x, y, counts)
        inf_diff = test_informative(x, y, counts)
        if label:
            lit_entails.append(lit_diff)
            inf_entails.append(inf_diff)
        else:
            lit_not_entails.append(lit_diff)
            inf_not_entails.append(inf_diff)
    return {
        "lit_entails": np.mean(lit_entails),
        "lit_not_entails": np.mean(lit_not_entails),
        "inf_entails": np.mean(inf_entails),
        "inf_not_entails": np.mean(inf_not_entails),
    }


counts = defaultdict(int)
metrics = defaultdict(list)
sample_rate = 10
min_sample = 100
n_data = []
for idx, line in enumerate(lines):
    counts[()] += 1
    if len(line) >= 1:
        counts[line[0],] += 1
        if len(line) >= 2:
            counts[line[0], line[1]] += 1
        else:
            counts[line[0], "E"] += 1
    else:
        counts["E",] += 1

    if idx % sample_rate == 0 and idx >= min_sample:
        for key, value in get_loss(counts).items():
            metrics[key].append(value)
        n_data.append(idx)

for key, values in metrics.items():
    plt.plot(n_data, values, label=key)
plt.xlabel("# data")
plt.ylabel("mean diff in log prob")
plt.title(f"Underlying speaker: {speaker}")
plt.legend()
plt.show()