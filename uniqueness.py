from itertools import product
import numpy as np

n_worlds = 4
n_utterances = 4

for bitstring in product(*[[0, 1] for _ in range(n_worlds * n_utterances)]):
    arr = np.array(list(bitstring))
    table = arr.reshape([n_worlds, n_utterances])
    utter_counts = arr.sum(axis=1, keepdims=True)
    breakpoint()