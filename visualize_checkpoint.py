"""
Visualize gradient evolution from training checkpoints
FIXED for PyTorch 2.6+
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize_gradients(checkpoint_path, save_dir=None):
    """Visualize gradient evolution from checkpoint."""

    if save_dir is None:
        save_dir = Path(checkpoint_path).parent / 'visualizations'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Load checkpoint - FIXED for PyTorch 2.6+
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract data
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    learning_rates = checkpoint.get('learning_rates', [])
    gradient_norms = checkpoint.get('gradient_norms', [])

    if not gradient_norms:
        print(" No gradient data found in checkpoint!")
        print("   This checkpoint was created with the old trainer.")
        print("   Use the ENHANCED trainer to track gradients.")
        print()
        print("To fix:")
        print("  1. cp pretrain_trainer_ENHANCED.py src1/training/pretrain_trainer.py")
        print("  2. Train a new model")
        print("  3. Run this script again")
        return

    print(f"  Found gradient data: {len(gradient_norms)} epochs")

    epochs = range(1, len(train_losses) + 1)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Gradient norm evolution
    axes[0, 0].plot(epochs[:len(gradient_norms)], gradient_norms, 
                   'purple', linewidth=2, alpha=0.7, label='Gradient Norm')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Average Gradient Norm', fontsize=12)
    axes[0, 0].set_title('Gradient Norm Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Add smoothed version
    if len(gradient_norms) >= 5:
        window = min(3, len(gradient_norms))
        smoothed = np.convolve(gradient_norms, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window, len(gradient_norms)+1), smoothed,
                       'orange', linewidth=2, alpha=0.8, label='Smoothed')
        axes[0, 0].legend()

    # 2. Gradient vs Loss
    axes[0, 1].scatter(gradient_norms, train_losses[:len(gradient_norms)], 
                      alpha=0.6, s=60, c=epochs[:len(gradient_norms)], cmap='viridis')
    axes[0, 1].set_xlabel('Gradient Norm', fontsize=12)
    axes[0, 1].set_ylabel('Train Loss', fontsize=12)
    axes[0, 1].set_title('Gradient Norm vs Training Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Epoch', fontsize=10)

    # 3. Gradient vs Learning Rate
    axes[1, 0].plot(epochs[:len(gradient_norms)], gradient_norms, 
                   'purple', label='Gradient Norm', linewidth=2)
    ax2 = axes[1, 0].twinx()
    ax2.plot(epochs[:len(learning_rates)], learning_rates, 
            'green', label='Learning Rate', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Gradient Norm', fontsize=12, color='purple')
    ax2.set_ylabel('Learning Rate', fontsize=12, color='green')
    axes[1, 0].set_title('Gradient Norm vs Learning Rate', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    ax2.set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Gradient stability (rolling std)
    if len(gradient_norms) >= 10:
        window = 5
        rolling_std = []
        for i in range(window, len(gradient_norms)):
            rolling_std.append(np.std(gradient_norms[i-window:i]))

        axes[1, 1].plot(range(window+1, len(gradient_norms)+1), rolling_std,
                       'red', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Gradient Std (rolling)', fontsize=12)
        axes[1, 1].set_title('Gradient Stability (Lower = More Stable)',
                           fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'Need more epochs\nfor stability analysis',
                       ha='center', va='center', fontsize=12,
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Gradient Stability', fontsize=14, fontweight='bold')

    plt.tight_layout()

    save_path = Path(save_dir) / 'gradient_evolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n Saved: {save_path}")
    plt.close()

    # Print statistics
    print("\n" + "="*70)
    print("GRADIENT STATISTICS")
    print("="*70)
    print(f"Initial gradient norm: {gradient_norms[0]:.6f}")
    print(f"Final gradient norm:   {gradient_norms[-1]:.6f}")
    print(f"Average gradient norm: {np.mean(gradient_norms):.6f}")
    print(f"Max gradient norm:     {np.max(gradient_norms):.6f}")
    print(f"Min gradient norm:     {np.min(gradient_norms):.6f}")
    print(f"Std gradient norm:     {np.std(gradient_norms):.6f}")
    print("="*70)

    # Interpretation
    print("\nINTERPRETATION:")
    print("="*70)
    avg_norm = np.mean(gradient_norms)
    if avg_norm < 1e-5:
        print("   VERY SMALL gradients - possible vanishing gradient problem")
    elif avg_norm > 10:
        print("   LARGE gradients - consider stronger gradient clipping")
    else:
        print("  Gradient magnitudes look healthy")

    # Check stability
    if len(gradient_norms) >= 10:
        recent_std = np.std(gradient_norms[-10:])
        if recent_std / np.mean(gradient_norms[-10:]) > 0.5:
            print("   High gradient variance - training might be unstable")
        else:
            print("  Gradients are stable")

    print("="*70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = r'D:\Thesis_new_v2\outputs\checkpoints\pretrain\checkpoint_best.pth'

    print("="*70)
    print("GRADIENT VISUALIZATION")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    visualize_gradients(checkpoint_path)