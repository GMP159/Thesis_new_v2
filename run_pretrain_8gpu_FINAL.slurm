#!/bin/bash
#SBATCH --job-name=hsi_pretrain_8gpu
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

# ===============================
# Environment setup
# ===============================
module load cuda/12.2

# Activate environment
source /scratch2/macu69pa/venv_macu69pa/bin/activate

# Move to project directory
cd /scratch2/macu69pa/hsi_project

# ===============================
# Runtime settings
# ===============================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# NCCL / PyTorch DDP settings
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Let SLURM manage GPU visibility
unset CUDA_VISIBLE_DEVICES

# PyTorch settings
export PYTHONUNBUFFERED=1

# Create directories
mkdir -p logs
mkdir -p checkpoints/pretrain

# ===============================
# Logging
# ===============================
echo "=========================================="
echo "HSI Cross-Domain Pretraining (DDP, 8 GPUs)"
echo "=========================================="
echo "Start time : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPUs       : 8"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Memory     : 128GB"
echo "Data Dir   : data_patches_32x32_split_zarr"
echo "Epochs     : 10"
echo "Batch Size : 64 per GPU (512 effective)"
echo "=========================================="

# Check CUDA
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
echo "=========================================="

# ===============================
# Launch training using DDP
# ===============================
torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  train_pretrain_DDP.py \
  --ddp \
  --data_root data_patches_32x32_split_zarr \
  --all_domains \
  --epochs 10 \
  --batch_size 64 \
  --num_workers 4 \
  --lr 2e-3 \
  --weight_decay 0.05 \
  --warmup_epochs 2 \
  --save_freq 2 \
  --log_gradients_every 10 \
  --save_dir checkpoints/pretrain

# Optional: Enable W&B (uncomment if you want live monitoring)
# Add these flags:
#  --use_wandb \
#  --wandb_project hsi-pretraining-8gpu \
#  --wandb_run_name run_10epochs_8gpu

# ===============================
# Completion log
# ===============================
echo "=========================================="
echo "Pretraining complete!"
echo "End time: $(date)"
echo "Logs: logs/pretrain_${SLURM_JOB_ID}.out"
echo "Checkpoints: checkpoints/pretrain/"
echo "=========================================="

# List checkpoints
echo "Generated checkpoints:"
ls -lh checkpoints/pretrain/
echo "=========================================="
