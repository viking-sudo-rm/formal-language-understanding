# synthetic-language-understanding

## Setup

```shell
pip install allennlp allennlp_models
```

## Usage

To generate training data:
```shell
mkdir data data/train
python generate.py true > data/train/true.txt
python generate.py rsa > data/train/rsa.txt
```

To generate a semantic evaluation suite to test with:
```shell
python generate.py eval > data/eval.tsv
```
Note that this doesn't depend on the pragmatics model used to generate the training data, just the semantics of the underlying language.

The following command will train and save models:
```shell
mkdir models
CUDA=0 TRAIN=data/train/true.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/true
CUDA=1 TRAIN=data/train/rsa.txt python -m allennlp train training_config/bi_lm.jsonnet -s=models/rsa
```
