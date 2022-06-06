#!/usr/bin/bash

outdir=$SCRATCH/slurm

for N in 10000 100000 1000000 10000000
do
    name=`numfmt --to=si $N`
    # --output=outdir/${lang}-${speaker}-${name}.out
    sbatch --job-name=${name}-${speaker::3} --export=ALL,l=$LANG,s=$SPEAKER,n=$N scripts/train.s
done
