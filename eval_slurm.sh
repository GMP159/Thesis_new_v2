#!/bin/bash
#SBATCH --job-name=hsi_eval
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ===============================
# Model Evaluation Script
# ===============================

# Activate environment
source /scratch2/macu69pa/venv_macu69pa/bin/activate

# Move to project directory
cd /scratch2/macu69pa/hsi_project

# Create directories
mkdir -p logs
mkdir -p evaluation_results

# ===============================
# Configuration
# ===============================
# IMPORTANT: Set your checkpoint path here
CHECKPOINT="checkpoints/pretrain/checkpoint_best.pth"

# Or use a specific epoch checkpoint:
# CHECKPOINT="checkpoints/pretrain/checkpoint_epoch010.pth"

# ===============================
# Logging
# ===============================
echo "=========================================="
echo "HSI Model Evaluation"
echo "=========================================="
echo "Start time : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "GPU        : 1"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Memory     : 32GB"
echo "Checkpoint : $CHECKPOINT"
echo "=========================================="

# Check CUDA
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Available checkpoints:"
    ls -lh checkpoints/pretrain/*.pth 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# ===============================
# Run Evaluation
# ===============================
echo "Starting evaluation..."
echo ""

python evaluate_pretrain.py \
  --checkpoint "$CHECKPOINT" \
  --data_root data_patches_32x32_split_zarr \
  --all_domains \
  --output_dir evaluation_results \
  --max_samples 5000

# ===============================
# Completion
# ===============================
echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "End time: $(date)"
echo "=========================================="

# List generated files
echo ""
echo "Generated evaluation files:"
ls -lh evaluation_results/ 2>/dev/null || echo "No files generated"
echo ""

# Display images info
echo "Visualization files:"
for img in evaluation_results/*.png; do
    if [ -f "$img" ]; then
        echo "  - $(basename $img)"
    fi
done
echo ""
echo "=========================================="
