# synthetic-language-understanding

## Setup

```shell
pip install allennlp allennlp_models
```

## Usage

To generate training data:
```shell
mkdir data data/quantifier
python generate.py > data/quantifier/rsa1.txt
python generate.py --depth=0 > data/quantifier/rsa0.txt
python generate.py --noisy > data/quantifier/rsa1-noisy.txt
python generate.py --dependent > data/quantifier/rsa1-dep.txt
python generate.py --dependent --depth=0 > data/quantifier/rsa0-dep.txt
```

The following command shows how to train and save models:
```shell
mkdir models models/quantifier
CUDA=0 TRAIN=data/quantifier/rsa1.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa1.txt
```
