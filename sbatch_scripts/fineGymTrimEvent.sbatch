#!/bin/bash
#SBATCH -p gpu-long
#SBATCH --mem=8192
#SBATCH --gres=gpu:2
#SBATCH --exclude=node53
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alalwani@cs.umass.edu
#SBATCH --job-name fineGymTrimEvent4
#SBATCH -o unity_logs/fineGymTrimEvent4.out
#SBATCH -e unity_logs/fineGymTrimEvent4.err
. /home/alalwani_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate open-mmlab-2
module add cuda/11.3.1
module add gcc/9.3.0
module list
srun python tools/data/gym/trim_event.py qGvIwgHzCH8 rrrgsW--AE8 s7pyZPf6fKU uDZVn-K4GE4 wYbfwx93VXk xHBUIt-airA