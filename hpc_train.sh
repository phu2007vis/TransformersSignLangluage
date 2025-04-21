#!/bin/bash
#SBATCH --job-name=Image_captioning
#SBATCH --partition=dgx-small
#SBATCH --time=24:00:00
#SBATCH --account=ddt_acc23

#SBATCH --ntasks-per-node=3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --export=MASTER_ADDR=localhost

# set up port to multi gpu
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
# load conda exe
source /home/21013187/anaconda3/etc/profile.d/conda.sh
squeue --me

cd /work/21013187/SAM-SLR-v2/phuoc_src
module load python cuda

#deactivate 3 times to assert no enviroment
conda deactivate
conda deactivate
conda deactivate
#activate env
conda activate py311

# CUDA SELECTION 
export CUDA_VISIBLE_DEVICES=1,2,6
python --version

torchrun --nproc_per_node=3 \
 --rdzv_id=100 \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$MASTER_ADDR:29400 \
 /work/21013187/SAM-SLR-v2/phuoc_src/train.py

