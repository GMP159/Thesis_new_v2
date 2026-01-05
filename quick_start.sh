#!/bin/bash
# Quick start script for pre-training

echo "================================"
echo "MaskedSST Pre-training"
echo "================================"
echo ""

# Test with 2 domains first (fast)
echo "Testing with 2 domains (apfel_swir + branches)..."
python train_pretrain.py \
  --data_root D:/Thesis_new/data_patches_32x32_split_zarr \
  --domains apfel_swir branches \
  --epochs 10 \
  --batch_size 64 \
  --save_dir checkpoints/pretrain_test

echo ""
echo "Test complete!"
echo ""
echo "To train on all domains for full 200 epochs:"
echo "  python train_pretrain.py --all_domains --epochs 200"
