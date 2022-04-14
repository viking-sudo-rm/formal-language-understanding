# synthetic-language-understanding

## Setup

```shell
pip install allennlp allennlp_models
```

To get a singularity container for `allennlp`, one can also do:

```shell
singularity pull docker://allennlp/allennlp:latest
```

## Usage

To generate training data (this can also be found in generate_data.sh):
```shell
mkdir data data/quantifier
python generate.py > data/quantifier/rsa1.txt
python generate.py --depth=0 > data/quantifier/rsa0.txt
python generate.py --noisy > data/quantifier/rsa1-noisy.txt
python generate.py --dependent > data/quantifier/rsa1-dep.txt
python generate.py --noisy --depth=0 > data/quantifier/rsa0-noisy.txt
```

The following command shows how to train and save models:
```shell
mkdir models models/quantifier
CUDA=0 TRAIN=data/quantifier/rsa1.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1
CUDA=0 TRAIN=data/quantifier/rsa0.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa0
CUDA=0 TRAIN=data/quantifier/rsa1-noisy.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1-noisy
CUDA=0 TRAIN=data/quantifier/rsa1-dep.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1-dep
CUDA=0 TRAIN=data/quantifier/rsa0-noisy.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa0-noisy
```
