# synthetic-language-understanding

## Setup

```shell
pip install allennlp allennlp_models
```

## Usage

To generate training data:
```shell
mkdir data data/quantifier
python generate.py true > data/quantifier/true.txt
python generate.py rsa > data/quantifier/rsa.txt
python generate.py eval > data/quantifier/eval.tsv
```

The following command will train and save models:
```shell
mkdir models models/quantifier
CUDA=0 TRAIN=data/quantifier/true.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/true
CUDA=1 TRAIN=data/quantifier/rsa.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/quantifier/rsa
```
