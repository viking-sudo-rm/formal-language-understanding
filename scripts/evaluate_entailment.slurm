#!/bin/bash
#SBATCH --job-name=generate
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -c 4


PROJECT_ROOT=/scratch/asw462/synthetic-language-understanding

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate synthetic_language_understanding

cd ${PROJECT_ROOT}

python evaluate_entailment.py \
--test_data=data/powerset/dependent/eval_entail-3_worlds-5_sents.tsv \
--distributional_model=${1} \
--lang=powerset \
--dependent \
--n_items=3 \
--temp=3 \
--cost=0.1 \
--training_dir=data/powerset/dependent \
--order=3 \
--size=${4} \
--plot_type ${2} \
--complexity=${3} \
--n_increments=22 \
--auc
"


