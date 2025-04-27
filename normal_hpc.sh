#!/bin/bash
#SBATCH --job-name=Image_captioning
#SBATCH --partition=normal
#SBATCH --time=6:00:00
#SBATCH --account=ddt_acc23
#SBATCH --mem=64G  

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3

#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err

source /home/21013187/anaconda3/etc/profile.d/conda.sh
# squeue --me
cd /work/21013187/SAM-SLR-v2/
module load python cuda
conda deactivate
conda deactivate
conda deactivate

conda activate py311



CUDA_VISIBLE_DEVICES=7 python /work/21013187/SAM-SLR-v2/phuoc_src/train.py 
# python /work/21013187/SAM-SLR-v2/phuoc_src/helper_fn/generate_cache.py
