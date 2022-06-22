
## Generate datasets including test data
```shell
for lang in powerset; do
  for agent in vanilla noisy dependent; do
    rm -rf data/${lang}/${agent}
    mkdir data/${lang}/${agent}
#    python generate.py ${lang} --${agent} --n_items=3 -n 1000000000 --temp=3 --cost=0.1 > data/${lang}/${agent}/1b.txt
#    python generate.py ${lang} --${agent} --n_items=3 -n 100000000 --temp=3 --cost=0.1 > data/${lang}/${agent}/100m.txt
#    python generate.py ${lang} --${agent} --n_items=3 -n 10000000 --temp=3 --cost=0.1 > data/${lang}/${agent}/10m.txt
#    python generate.py ${lang} --${agent} --n_items=3 -n 1000000 --temp=3 --cost=0.1 > data/${lang}/${agent}/1m.txt
#    python generate.py ${lang} --${agent} --n_items=3 -n 100000 --temp=3 --cost=0.1 > data/${lang}/${agent}/100k.txt
    python generate.py ${lang} --${agent} --n_items=3 -n 10000 --temp=3 --cost=0.1 > data/${lang}/${agent}/10000.txt
    python generate.py ${lang} --${agent} --n_items=3 -n 1000 --temp=3 --cost=0.1 > data/${lang}/${agent}/1000.txt
    python generate.py ${lang} --${agent} --n_items=3 -n 100 --temp=3 --cost=0.1 > data/${lang}/${agent}/100.txt
#    python generate_compositional_test_data.py ${lang} --${agent} --n_items=3 --temp=3 --cost=0.1 --max_sent_len=5 --eval_dir=data/${lang}/${agent}
#    CUDA=0 TRAIN=data/${lang}/${agent}/train.txt DEV=data/${lang}/${agent}/dev.txt N_EPOCHS=20 python -m allennlp train training_config/bi_lm.jsonnet -s=models/${lang}/${agent}.txt
  done
done
```





# Compute AUC with RSA Model
```shell
cd src/evaluation
python evaluate_entailment.py --test_data=../../data/powerset/dependent/eval_entail-3_worlds-5_sents.tsv \
--distributional_model=RSA \
--lang=powerset \
--dependent \
--n_items=3 \
--temp=3 \
--cost=0.1

```


## Set general variables
```shell
export PROJECT_ROOT=/scratch/asw462/synthetic-language-understanding
export lang=powerset
export temp=3
export cost=0.1
export n_items=3
export n=1000000
```

## Generate Training Data
Run this to generate training data. You can also loop over it to launch multiple jobs in parallel, as I did.
```shell
cd ${PROJECT_ROOT}
for agent in vanilla dependent; do
  rm -rf data/${lang}/${agent}
  mkdir data/${lang}/${agent}
  python generate.py ${lang} \
                        --$2 \
                        --n_items=${n_items} \
                        -n ${n} \
                        --temp=${temp} \
                        --cost=${cost} \
    > data/${lang}/$2/${n}.txt
done
```

## Train Neural LMs
```shell
cd ${PROJECT_ROOT}
CUDA=0 
N_EPOCHS=20 
for agent in vanilla dependent; do
  TRAIN=data/${lang}/${agent}/10000.txt 
  DEV=data/${lang}/${agent}/1000.txt 
  python -m allennlp train training_config/bi_lm.jsonnet -s=models/${lang}/${agent}.txt
done
```

## Generate Test Data
```shell
cd ${PROJECT_ROOT}
for agent in vanilla dependent; do
  python generate_compositional_test_data.py ${lang} 
    --${agent} \
    --n_items=3 \
    --temp=3 \
    --cost=0.1 \
    --max_sent_len=5 \
    --eval_dir=data/powerset/dependent
done
```