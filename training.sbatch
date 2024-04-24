#!/bin/bash
#
#SBATCH --job-name=hf_diff_apf_training
#SBATCH --output=cluster/hf_diff_apf_training_output.txt
#SBATCH --error=cluster/hf_diff_apf_training_error.txt
#
#SBATCH --time=96:00:00
#SBATCH --chdir=/home/hf1/Documents/RLAutoAlign

module purge
module load anaconda3
module load cuda-11.2.1
conda init bash
# source activate hf_diff_apf
conda activate hf_diff_apf

srun python stablebaseline_PPO.py