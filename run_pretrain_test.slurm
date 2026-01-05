#!/bin/bash
#SBATCH --job-name=hsi_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "Quick test started at: $(date)"

module purge
module load Python/3.11
module load CUDA/12.1

source ~/miniconda3/bin/activate
conda activate thesis_env

cd $HOME/Thesis_new_v2

# Quick 3-epoch test
python train_pretrain.py \
    --domains branches apfel_swir \
    --epochs 3 \
    --batch_size 8 \
    --num_workers 4

echo "Test finished at: $(date)"
