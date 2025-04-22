#!/bin/bash
#SBATCH --job-name=Image_captioning
#SBATCH --partition=dgx-small
#SBATCH --time=24:00:00
#SBATCH --account=ddt_acc23

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err

source /home/21013187/anaconda3/etc/profile.d/conda.sh
squeue --me
cd /work/21013187/SAM-SLR-v2/
module load python cuda
conda deactivate
conda deactivate
conda deactivate

conda activate py311

python --version

CUDA_VISIBLE_DEVICES=6,7  python /work/21013187/SAM-SLR-v2/phuoc_src/train.py 

