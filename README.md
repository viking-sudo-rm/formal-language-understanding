# synthetic-language-understanding

To generate data:
```shell
python generate.py --only_true > data/only_true.txt
```

To train:
```shell
TRAIN=data/only_true.txt python -m allennlp train training_config/bi_lm.jsonnet -s=/tmp/only_true
```
