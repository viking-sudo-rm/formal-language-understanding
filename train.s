#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
# we expect the job to use no more than 2GB of memory:
#SBATCH --mem=16GB
#SBATCH --job-name=trainLm
#SBATCH --mail-type=END
#SBATCH --mail-user=willm@nyu.edu
#SBATCH --output=/scratch/wcm9940/slurm/%j.out
#SBATCH --gres=gpu:1

module purge
module load python3.8

conda activate $SCRATCH/.conda/bin/allennlp
dir=$SCRATCH/synthetic-language-understanding
train=$dir/data/powerset/rsa1.txt
model=$dir/models/powerset/rsa1

CUDA=0 TRAIN=$train N_EPOCHS=3 allennlp train training_config/bi_lm.jsonnet -s=$model
