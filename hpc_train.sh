#!/bin/bash
#SBATCH --job-name=trans_lately
#SBATCH --partition=dgx-small
#SBATCH --time=48:00:00
#SBATCH --account=ddt_acc23

#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --export=MASTER_ADDR=localhost

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
source /home/21013187/anaconda3/etc/profile.d/conda.sh
squeue --me
cd /work/21013187/SAM-SLR-v2/phuoc_src
module load python cuda
conda deactivate
conda deactivate
conda deactivate

conda activate py311
export CUDA_VISIBLE_DEVICES=3,4
python --version

torchrun --nproc_per_node=2 \
 --rdzv_id=100 \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$MASTER_ADDR:29401 \
 /work/21013187/SAM-SLR-v2/phuoc_src/train.py --config_path=/work/21013187/SAM-SLR-v2/phuoc_src/config/landmarks.yaml \
 --save_name='trans_lately'

