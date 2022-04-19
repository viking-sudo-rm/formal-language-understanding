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

lang="powerset"
temp=5
cost=.5

# DATA=$DIR/data
DATA=$SCRATCH/synthetic-language-understanding/data
mkdir $DATA/$lang
mkdir $DATA/$lang/literal
mkdir $DATA/$lang/informative
mkdir $DATA/$lang/independent

# for n in 10000 100000 1000000 10000000
for n in 1000000 10000000
do
    echo Sampling with n=$n...
    python generate.py $lang -n=$n --temp=$temp --cost=$cost  > $DATA/$lang/independent/$n.txt
    python generate.py $lang -n=$n --temp=$temp --cost=$cost --depth=0 > $DATA/$lang/literal/$n.txt
    python generate.py $lang -n=$n --temp=$temp --cost=$cost --dependent > $DATA/$lang/informative/$n.txt
#    python generate.py $lang -n=$n --temp=5 --cost=$cost --noisy > data/$lang/rsa1-noisy.txt
#    python generate.py $lang -n=$n --temp=5 --cost=$cost --noisy --depth=0 > data/$lang/rsa0-noisy.txt
done

# Generate dev set using a different random seed.
echo Sampling dev sets with n=$n...
python generate.py $lang --seed=3 -n=10000 --temp=$temp --cost=$cost > $DATA/$lang/independent/dev.txt
python generate.py $lang --seed=3 -n=10000 --temp=$temp --cost=$cost --depth=0 > $DATA/$lang/literal/dev.txt
python generate.py $lang --seed=3 -n=10000 --temp=$temp --cost=$cost --dependent > $DATA/$lang/informative/dev.txt

# Generate test set using a different random seed.
echo Sampling test sets with n=$n...
python generate.py $lang --seed=4 -n=10000 --temp=$temp --cost=$cost > $DATA/$lang/independent/test.txt
python generate.py $lang --seed=4 -n=10000 --temp=$temp --cost=$cost --depth=0 > $DATA/$lang/literal/test.txt
python generate.py $lang --seed=4 -n=10000 --temp=$temp --cost=$cost --dependent > $DATA/$lang/informative/test.txt
