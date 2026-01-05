#!/bin/bash
#SBATCH --job-name=hsi_pretrain_wandb
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@ovgu.de

echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Load modules
module purge
module load Python/3.11
module load CUDA/12.1
module load cuDNN/8.9

# Activate environment
source ~/miniconda3/bin/activate
conda activate thesis_env

# Set environment
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=$(cat ~/.wandb_api_key)  # Store API key here
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd $HOME/Thesis_new_v2
mkdir -p logs

# Run with W&B
python train_pretrain.py \
    --domains branches apfel_swir grapes pottery \
    --epochs 50 \
    --batch_size 32 \
    --num_workers 8 \
    --learning_rate 1e-4 \
    --warmup_epochs 5 \
    --weight_decay 0.05 \
    --save_every 5 \
    --log_gradients_every 10 \
    --mixed_precision \
    --compile_model \
    --use_wandb \
    --wandb_project "hsi-pretrain" \
    --wandb_name "pretrain_4domains_50epochs"

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
