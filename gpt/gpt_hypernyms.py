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

context_template = "John is a {noun1}. Thus, "
full_template = "John is a {noun1}. Thus, John is a {noun2}."


nouns = ["sailor", "human", "cat", "animal"]


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


def contextualize(noun1, noun2):
    context = context_template.format(noun1=noun1, noun2=noun2)
    full = full_template.format(noun1=noun1, noun2=noun2)
    return full, context


@lru_cache(None)
def get_joint_probability(quant1, quant2, total=10):
    running_sum = 0.
    for noun in nouns:
        full1, context1 = contextualize(noun, quant1)
        prob1 = get_conditional_probability(full1, context1)
        full2, context2 = contextualize(noun, quant2)
        prob2 = get_conditional_probability(full2, context2)
        running_sum += prob1 * prob2
    return running_sum / total


def get_entailment_score(quant1, quant2):
    print("Doing", quant1, quant2)
    pxy = get_joint_probability(quant1, quant2)
    pxx = get_joint_probability(quant1, quant1)
    px = get_joint_probability(quant1, "")
    py = get_joint_probability(quant2, "")
    return abs(px * pxy - py * pxx)


matrix = np.array([[get_entailment_score(x, y) for y in nouns] for x in nouns])

true_matrix = np.ones([len(nouns), len(nouns)])
true_matrix[np.arange(len(nouns)), np.arange(len(nouns))] = 0
true_matrix[0, 1] = 0
true_matrix[2, 3] = 0

corr = np.corrcoef(matrix.flatten(), true_matrix.flatten())
print("Correlation", corr)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.matshow(matrix)
ax2.matshow(true_matrix)
plt.tight_layout()
plt.show()
