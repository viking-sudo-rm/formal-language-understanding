# Formal Language Understanding

Codebase for investigating the semantic capabilities of language models in formal language environments. Specifically, in these environments, the process generating the training data is idealized in terms of pragmatic theory and controllable via many hyperparameters.

## Setup

```shell
pip install allennlp allennlp_models
```

To get a singularity container for `allennlp`, one can also do:

```shell
singularity pull docker://allennlp/allennlp:latest
```

## Generate Synthetic Data

To generate training data in the language `powerset`:

```shell
python generate.py powerset --temp=5 --cost=.5  > documents.txt
```

Full documentation of all the training data can be found in generate.s.

## Train LMs

The following command shows how to train and save models:
```shell
mkdir models models/quantifier
CUDA=0 TRAIN=data/quantifier/rsa1.txt allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1
CUDA=0 TRAIN=data/quantifier/rsa0.txt allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa0
CUDA=0 TRAIN=data/quantifier/rsa1-noisy.txt allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1-noisy
CUDA=0 TRAIN=data/quantifier/rsa1-dep.txt allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1-dep
CUDA=0 TRAIN=data/quantifier/rsa0-noisy.txt allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa0-noisy
```
