#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=train
#SBATCH --mail-type=END
#SBATCH --mail-user=willm@nyu.edu
#SBATCH --output=/scratch/wcm9940/slurm/%j.out
#SBATCH --gres=gpu:1

# cf. https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/conda-environments-python-r#h.p_ID_196
module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

myconda=/scratch/wcm9940/.conda/envs/allennlp
conda activate $myconda
export PATH=$myconda/bin:$PATH

# l=powerset
# s=literal
# n=100000

dir=$SCRATCH/synthetic-language-understanding
train=$dir/data/$l/$s/$n.txt
dev=$dir/data/$l/$s/dev.text

mkdir -p $dir/models/$l/$s
CUDA=0 TRAIN=$train DEV=$dev N_EPOCHS=5 python -m allennlp train training_config/bi_lm.jsonnet -s=$dir/models/$l/$s/$n
