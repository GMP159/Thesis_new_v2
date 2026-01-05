"""
Quick check of training progress and checkpoint quality
Run this locally or on cluster to inspect your trained model
"""

import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def check_checkpoint(checkpoint_path):
    """Quick inspection of a checkpoint."""
    print(f"\n{'='*80}")
    print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Basic info
        print(f"\n1. Basic Information:")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Best Val Loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
        
        # Training history
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            print(f"\n2. Training History:")
            print(f"   Epochs trained: {len(train_losses)}")
            print(f"   Final train loss: {train_losses[-1]:.4f}")
            print(f"   Final val loss: {val_losses[-1]:.4f}")
            print(f"   Best val loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses)) + 1})")
            
            # Check for overfitting
            gap = train_losses[-1] - val_losses[-1]
            print(f"   Train-Val gap: {gap:.4f}")
            if abs(gap) < 0.05:
                print(f"     Good generalization")
            elif gap < -0.1:
                print(f"     Validation better than train (possible issue)")
            else:
                print(f"     Possible overfitting")
        
        # Gradient norms
        if 'gradient_norms' in checkpoint:
            grad_norms = checkpoint['gradient_norms']
            if len(grad_norms) > 0:
                print(f"\n3. Gradient Statistics:")
                print(f"   Average gradient norm: {np.mean(grad_norms):.6f}")
                print(f"   Max gradient norm: {np.max(grad_norms):.6f}")
                print(f"   Min gradient norm: {np.min(grad_norms):.6f}")
                
                # Check for gradient issues
                if np.mean(grad_norms) < 1e-6:
                    print(f"     Very small gradients (possible vanishing)")
                elif np.mean(grad_norms) > 10:
                    print(f"     Large gradients (possible exploding)")
                else:
                    print(f"     Healthy gradient flow")
        
        # Learning rate schedule
        if 'learning_rates' in checkpoint:
            lrs = checkpoint['learning_rates']
            print(f"\n4. Learning Rate:")
            print(f"   Initial: {lrs[0]:.2e}")
            print(f"   Final: {lrs[-1]:.2e}")
            print(f"   Decay ratio: {lrs[-1]/lrs[0]:.2%}")
        
        # Model size
        state_dict = checkpoint['model_state_dict']
        n_params = sum(p.numel() for p in state_dict.values())
        print(f"\n5. Model:")
        print(f"   Parameters: {n_params:,}")
        print(f"   Size: {n_params * 4 / 1e6:.2f} MB")
        
        return checkpoint
        
    except Exception as e:
        print(f"     Error loading checkpoint: {e}")
        return None


def plot_training_curves(checkpoint):
    """Plot training curves from checkpoint."""
    if checkpoint is None:
        return
    
    try:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        lrs = checkpoint['learning_rates']
        grad_norms = checkpoint.get('gradient_norms', [])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(epochs, lrs, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        if len(grad_norms) > 0:
            axes[1, 0].plot(epochs[:len(grad_norms)], grad_norms, 'purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Evolution')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Loss improvement
        loss_improvement = [val_losses[0]] + [val_losses[i-1] - val_losses[i] for i in range(1, len(val_losses))]
        axes[1, 1].plot(epochs, loss_improvement, 'orange', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss Improvement')
        axes[1, 1].set_title('Per-Epoch Validation Improvement')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n  Saved training curves to: training_analysis.png")
        plt.close()
        
    except Exception as e:
        print(f"  Error plotting curves: {e}")


def compare_checkpoints(checkpoint_dir):
    """Compare all checkpoints in a directory."""
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pth')))
    
    if len(checkpoint_files) == 0:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Found {len(checkpoint_files)} checkpoints")
    print(f"{'='*80}\n")
    
    results = []
    for cp_path in checkpoint_files:
        cp = torch.load(cp_path, map_location='cpu', weights_only=False)
        results.append({
            'file': os.path.basename(cp_path),
            'epoch': cp.get('epoch', 0),
            'train_loss': cp['train_losses'][-1] if 'train_losses' in cp else 0,
            'val_loss': cp['val_losses'][-1] if 'val_losses' in cp else 0,
            'best_val': cp.get('best_val_loss', float('inf'))
        })
    
    # Print comparison table
    print(f"{'Checkpoint':<30} {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Best Val':<12}")
    print(f"{'-'*85}")
    for r in results:
        print(f"{r['file']:<30} {r['epoch']:<8} {r['train_loss']:<12.4f} {r['val_loss']:<12.4f} {r['best_val']:<12.4f}")
    
    # Best checkpoint
    best = min(results, key=lambda x: x['best_val'])
    print(f"\n  Best checkpoint: {best['file']}")
    print(f"  Val loss: {best['best_val']:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Quick checkpoint analysis')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint to analyze')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pretrain',
                       help='Directory with checkpoints')
    parser.add_argument('--plot', action='store_true',
                       help='Generate training curve plots')
    args = parser.parse_args()
    
    if args.checkpoint:
        # Analyze specific checkpoint
        cp = check_checkpoint(args.checkpoint)
        if args.plot and cp is not None:
            plot_training_curves(cp)
    else:
        # Compare all checkpoints
        results = compare_checkpoints(args.checkpoint_dir)
        
        # Plot best checkpoint
        if args.plot and results:
            best = min(results, key=lambda x: x['best_val'])
            best_path = os.path.join(args.checkpoint_dir, best['file'])
            cp = torch.load(best_path, map_location='cpu', weights_only=False)
            plot_training_curves(cp)


if __name__ == '__main__':
    main()
