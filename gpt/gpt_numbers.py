import torch
from transformers import GPTNeoForCausalLM, GPTNeoConfig, GPT2Tokenizer
from functools import lru_cache
import numpy as np
from tqdm import tqdm


ckpt = "EleutherAI/gpt-neo-1.3B"
model = GPTNeoForCausalLM(GPTNeoConfig.from_pretrained(ckpt))
model.cuda()
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained(ckpt)

pairs = [
    ("dogs", "animals"),
    ("cats", "animals"),
    ("books", "things"),
    ("kayaks", "boats"),
]

context_template = "There are {total} {noun_general}. {number} of them are {noun_subset}. Thus, "
full_template = "There are {total} {noun_general}. {number} of them are {noun_subset}. Thus, {quantifier} of the {noun_general} are {noun_subset}."


@lru_cache(None)
@torch.no_grad()
def get_conditional_probability(full, context):
    context_tokens = tokenizer(context, return_tensors="pt").input_ids
    context_tokens = context_tokens.cuda()
    context_out = model(input_ids=context_tokens, labels=context_tokens)

    full_tokens = tokenizer(full, return_tensors="pt").input_ids
    full_tokens = full_tokens.cuda()
    full_out = model(input_ids=full_tokens, labels=full_tokens)

    return 2 ** (full_out["loss"] - context_out["loss"]).item()


def contextualize(quantifier, number, total, noun_general, noun_subset):
    number = str(number)
    total = str(total)
    context = context_template.format(total=total, number=number, noun_general=noun_general, noun_subset=noun_subset)
    full = full_template.format(total=total, number=number, quantifier=quantifier, noun_general=noun_general, noun_subset=noun_subset)
    return full, context


@lru_cache(None)
def get_joint_probability(quant1, quant2, total=10):
    running_sum = 0.
    for number in range(total + 1):
        for noun_subset, noun_general in tqdm(pairs):
            full1, context1 = contextualize(quant1, number, total, noun_general, noun_subset)
            prob1 = get_conditional_probability(full1, context1)
            full2, context2 = contextualize(quant2, number, total, noun_general, noun_subset)
            prob2 = get_conditional_probability(full2, context2)
            running_sum += prob1 * prob2
    return running_sum / (total * len(pairs))


def get_entailment_score(quant1, quant2):
    print("Doing", quant1, quant2)
    pxy = get_joint_probability(quant1, quant2)
    pxx = get_joint_probability(quant1, quant1)
    px = get_joint_probability(quant1, "")
    py = get_joint_probability(quant2, "")
    return abs(px * pxy - py * pxx)


quantifiers = ["one", "two", "three", "four", "five"]
matrix = np.array([[get_entailment_score(x, y) for y in quantifiers] for x in quantifiers])
true_matrix = np.expand_dims(np.arange(5), -1) > np.expand_dims(np.arange(5), 0)
corr = np.corrcoef(matrix.flatten(), true_matrix.flatten())
print("Correlation", corr)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.matshow(matrix)
ax2.matshow(true_matrix)
plt.tight_layout()
plt.show()
