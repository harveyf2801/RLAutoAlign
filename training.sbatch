#!/bin/bash
#
#SBATCH --job-name=hf_diff_apf_training
#SBATCH --output=hf_diff_apf_training_output.txt
#SBATCH --error=hf_diff_apf_training_error.txt
#
#SBATCH --time=15:00:00
#SBATCH --chdir=/home/hf1/Documents/RLAutoAlign

module purge
module load anaconda3
module load cuda-11.2.1
#conda init bash
#conda activate hf_diff_apf
source activate hf_diff_apf

srun python stablebaseline_PPO.py