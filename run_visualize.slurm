#!/bin/bash
#SBATCH --job-name=viz_pretrain
#SBATCH --partition=p0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err

# ===============================
# HSI Pretrained Model Visualization
# ===============================

echo "=========================================="
echo "HSI Model Visualization"
echo "=========================================="
echo "Start time : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "=========================================="

# Activate environment
source /scratch2/macu69pa/venv_macu69pa/bin/activate

# Move to project directory
cd /scratch2/macu69pa/hsi_project

# Create logs directory
mkdir -p logs

# Check CUDA
echo "Checking CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo "=========================================="

# Run visualization
echo "Running visualization script..."
python visualize_pretrain_fixed.py

# Check outputs
echo "=========================================="
echo "Generated visualizations:"
ls -lh checkpoints/pretrain/visualizations/
echo "=========================================="

echo "Complete! $(date)"
echo "Output files in: checkpoints/pretrain/visualizations/"
echo "=========================================="
