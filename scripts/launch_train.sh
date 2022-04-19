#!/usr/bin/bash

outdir=$SCRATCH/slurm

for N in 10000 100000 1000000 10000000
do
    name=${N::-3}k
    sbatch --job-name=$name --output=outdir/${lang}-${speaker}-${name}.out --export=ALL,l=$LANG,s=$SPEAKER,n=$N train.s
done
