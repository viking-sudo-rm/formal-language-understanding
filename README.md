# Formal Language Understanding

Codebase for investigating the semantic capabilities of language models in formal language environments. Specifically, in these environments, the process generating the training data is idealized in terms of pragmatic theory and controllable via many hyperparameters.

## Setup

```shell
pip install allennlp allennlp_models
```

<!-- To get a singularity container for `allennlp`, one can also do:

```shell
singularity pull docker://allennlp/allennlp:latest
``` -->

## Generate Synthetic Data

To generate training data in the language `powerset`:

```shell
python generate.py powerset --temp=5 --cost=.5  > documents.txt
```

Development and testing sets can be generated by specifying a different random seeds.

```shell
python generate.py powerset --seed=3 --temp=5 --cost=.5  > dev_documents.txt
```

Full documentation of all the training data can be found in generate.s.

## Train LMs

The following command shows how to train and save a language model on the synthetic data:
```shell
mkdir models models/quantifier
CUDA=0 TRAIN=documents.txt DEV=dev_documents.txt allennlp train training_config/bi_lm.jsonnet -s=rsa1_model
```

Note that this part can be done using whatever language modeling framework you want, but I'm using AllenNLP.
