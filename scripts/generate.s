#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=generate.s
#SBATCH --mail-type=END
#SBATCH --mail-user=willm@nyu.edu
#SBATCH --output=/scratch/wcm9940/slurm/%j.out

# cf. https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/conda-environments-python-r#h.p_ID_196
module purge;
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;

myconda=/scratch/wcm9940/.conda/envs/allennlp
conda activate $myconda
export PATH=$myconda/bin:$PATH

lang="powerset"
n_items=3

# DATA=$DIR/data
DATA=$SCRATCH/synthetic-language-understanding/data
mkdir $DATA/${lang}-${n_items}
mkdir $DATA/${lang}-${n_items}/literal
mkdir $DATA/${lang}-${n_items}/informative
mkdir $DATA/${lang}-${n_items}/independent

for n in 10000 100000 1000000 10000000
# for n in 1000000 10000000
# for n in
do
    echo Sampling with n=$n...
    python generate.py $lang --n_items=$n_items -n=$n --temp=3 --cost=0.5  > $DATA/${lang}-${n_items}/independent/$n.txt
    python generate.py $lang --n_items=$n_items -n=$n --temp=3 --cost=0.5 --depth=0 > $DATA/${lang}-${n_items}/literal/$n.txt
    python generate.py $lang --n_items=$n_items -n=$n --temp=3 --cost=0.1 --dependent > $DATA/${lang}-${n_items}/informative/$n.txt
done

# FIXME: For some reason, these lines don't work. I manually entered them in console to generate dev/test.

# Generate dev set using a different random seed.
echo "Sampling dev sets..."
python generate.py $lang --n_items=$n_items --seed=3 -n=10000 --temp=3 --cost=0.5 > $DATA/${lang}-${n_items}/independent/dev.txt
python generate.py $lang --n_items=$n_items --seed=3 -n=10000 --temp=3 --cost=0.5 --depth=0 > $DATA/${lang}-${n_items}/literal/dev.txt
python generate.py $lang --n_items=$n_items --seed=3 -n=10000 --temp=3 --cost=0.1 --dependent > $DATA/${lang}-${n_items}/informative/dev.txt

# Generate test set using a different random seed.
echo "Sampling test sets..."
python generate.py $lang --n_items=$n_items --seed=4 -n=10000 --temp=3 --cost=0.5 > $DATA/${lang}-${n_items}/independent/test.txt
python generate.py $lang --n_items=$n_items --seed=4 -n=10000 --temp=3 --cost=0.5 --depth=0 > $DATA/${lang}-${n_items}/literal/test.txt
python generate.py $lang --n_items=$n_items --seed=4 -n=10000 --temp=3 --cost=0.1 --dependent > $DATA/${lang}-${n_items}/informative/test.txt
