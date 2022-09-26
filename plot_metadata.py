import matplotlib.pyplot as plt
from collections import Counter


with open("data/powerset/dependent/10000.txt") as f:
    texts = [l.split() for l in f]

lengths = [len(t) for t in texts]
utterances = [u for t in texts for u in t]
c = Counter(utterances)
V = ['111', '110', '101', '011', '100', '010', '001']
total = len(utterances)

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(8, 2.5)

axs[0].bar(x=V, height=[c[v]/total for v in V])
axs[1].hist(lengths, bins=range(1, 11), density=True)

axs[0].set_xlabel("Utterance")
axs[0].set_ylabel("Relative frequency")
axs[1].set_xlabel("Text length")
axs[1].set_ylabel("Relative frequency")

plt.tight_layout()
plt.savefig("plots/metadata.pdf")
