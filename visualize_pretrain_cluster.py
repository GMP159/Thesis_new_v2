"""
FIXED Visualization script - For CLUSTER
"""

import matplotlib
matplotlib.use('Agg')  # ← For headless server

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
import zarr
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src1.models.masked_sst import create_model
from src1.data.dataset import create_dataloaders


def infer_num_classes(checkpoint_path):
    """Infer number of classes from checkpoint - handles DDP."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Try both with and without 'module.' prefix (DDP)
    possible_keys = [
        'module.classification_head.classifier.weight',  # DDP
        'classification_head.classifier.weight'          # Non-DDP
    ]

    for key in possible_keys:
        if key in checkpoint['model_state_dict']:
            weight_shape = checkpoint['model_state_dict'][key].shape
            num_classes = weight_shape[0]
            return num_classes

    return 8  # Default to 8


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint - handles DDP."""
    print(f"Loading checkpoint: {checkpoint_path}")

    num_classes = infer_num_classes(checkpoint_path)
    print(f"  Detected num_classes: {num_classes}")

    model = create_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle DDP: remove 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("  Removing 'module.' prefix from DDP checkpoint...")
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    history = {
        'epoch': checkpoint.get('epoch', 0),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'learning_rates': checkpoint.get('learning_rates', []),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
    }

    print(f"  Epoch: {history['epoch']}")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")

    return model, history


def plot_training_curves(history, save_path):
    """Plot training curves."""
    print("\nPlotting training curves...")

    train_losses = history['train_losses']
    val_losses = history['val_losses']
    lrs = history['learning_rates']

    if len(train_losses) == 0:
        print("  No training history found!")
        return

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    axes[0, 0].axhline(y=history['best_val_loss'], color='g', linestyle='--', 
                       label=f'Best: {history["best_val_loss"]:.4f}')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Loss (log scale)
    axes[0, 1].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, val_losses, 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[0, 1].set_title('Loss (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 0].plot(epochs, lrs, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Improvement
    if len(val_losses) > 1:
        improvement = [(val_losses[0] - v) / val_losses[0] * 100 for v in val_losses]
        axes[1, 1].plot(epochs, improvement, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Improvement (%)', fontsize=12)
        axes[1, 1].set_title('Val Loss Improvement', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        final_improvement = improvement[-1]
        axes[1, 1].text(0.05, 0.95, f'Total: {final_improvement:.1f}%', 
                       transform=axes[1, 1].transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def sample_from_each_domain(data_loader, domain_mapping, samples_per_domain=2):
    """Sample patches from each domain."""
    print("\nSampling from each domain...")

    domain_samples = {domain_id: [] for domain_id in domain_mapping.values()}

    for patches, labels in data_loader:
        for i in range(len(labels)):
            domain_id = labels[i].item()
            if len(domain_samples[domain_id]) < samples_per_domain:
                domain_samples[domain_id].append((patches[i], domain_id))

        if all(len(samples) >= samples_per_domain for samples in domain_samples.values()):
            break

    all_samples = []
    for domain_id in sorted(domain_samples.keys()):
        all_samples.extend(domain_samples[domain_id])

    print(f"  Collected {len(all_samples)} samples across {len(domain_samples)} domains")

    return all_samples


def visualize_reconstructions(model, data_loader, device, save_path, domain_mapping, n_samples=8):
    """Visualize reconstructions from ALL domains."""
    print("\nVisualizing reconstructions...")

    model = model.to(device)
    model.eval()

    id_to_domain = {v: k for k, v in domain_mapping.items()}

    samples_per_domain = max(2, n_samples // len(domain_mapping))
    all_samples = sample_from_each_domain(data_loader, domain_mapping, samples_per_domain)
    all_samples = all_samples[:n_samples]

    patches = torch.stack([s[0] for s in all_samples]).to(device)
    labels_np = np.array([s[1] for s in all_samples])

    with torch.no_grad():
        reconstruction, mask = model(patches, mode='reconstruction')

    patches_cpu = patches.cpu().numpy()
    mask_cpu = mask.cpu().numpy()

    fig, axes = plt.subplots(len(all_samples), 4, figsize=(16, len(all_samples) * 3))

    if len(all_samples) == 1:
        axes = axes.reshape(1, -1)

    for i in range(len(all_samples)):
        domain_name = id_to_domain.get(labels_np[i], f'Domain {labels_np[i]}')

        # Original (RGB)
        rgb_bands = [50, 100, 150]
        rgb_orig = patches_cpu[i, :, :, rgb_bands]
        if rgb_orig.shape[0] == 3:
            rgb_orig = np.transpose(rgb_orig, (1, 2, 0))
        rgb_orig = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)

        axes[i, 0].imshow(rgb_orig)
        axes[i, 0].set_title(f'Original: {domain_name}', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Mask
        mask_vis = np.kron(mask_cpu[i], np.ones((2, 2)))
        axes[i, 1].imshow(mask_vis, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Mask ({mask_cpu[i].mean()*100:.1f}%)', fontsize=10)
        axes[i, 1].axis('off')

        # Spectral signature
        center_h, center_w = 16, 16
        orig_spectrum = patches_cpu[i, center_h, center_w, :]

        axes[i, 2].plot(orig_spectrum, 'b-', alpha=0.8, linewidth=1.5)
        axes[i, 2].set_title('Center Spectrum', fontsize=10)
        axes[i, 2].set_xlabel('Band', fontsize=9)
        axes[i, 2].set_ylabel('Reflectance', fontsize=9)
        axes[i, 2].grid(True, alpha=0.3)

        # Average spectrum
        avg_spectrum = patches_cpu[i].mean(axis=(0, 1))
        axes[i, 3].plot(avg_spectrum, 'g-', alpha=0.8, linewidth=1.5)
        axes[i, 3].set_title('Avg Spectrum', fontsize=10)
        axes[i, 3].set_xlabel('Band', fontsize=9)
        axes[i, 3].set_ylabel('Reflectance', fontsize=9)
        axes[i, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def analyze_embeddings(model, data_loader, device, save_path, domain_mapping, n_samples=400):
    """t-SNE with ALL domains."""
    print("\nAnalyzing embeddings...")

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  scikit-learn not installed!")
        return

    model = model.to(device)
    model.eval()

    id_to_domain = {v: k for k, v in domain_mapping.items()}

    embeddings = []
    domain_labels = []

    print(f"  Extracting features from up to {n_samples} samples...")
    with torch.no_grad():
        total_samples = 0
        for patches, labels in tqdm(data_loader, desc='  Processing'):
            patches = patches.to(device)

            features, _ = model.forward_encoder(patches, apply_masking=False)
            pooled = features.mean(dim=(1, 2))

            embeddings.append(pooled.cpu())
            domain_labels.append(labels.cpu())

            total_samples += patches.size(0)
            if total_samples >= n_samples:
                break

    embeddings = torch.cat(embeddings, dim=0)[:n_samples].numpy()
    domain_labels = torch.cat(domain_labels, dim=0)[:n_samples].numpy()

    print(f"  Unique domains in sample: {np.unique(domain_labels)}")
    print(f"  Running t-SNE on {embeddings.shape[0]} samples...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 2 - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    unique_domains = np.unique(domain_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))

    for i, domain_id in enumerate(unique_domains):
        mask = domain_labels == domain_id
        domain_name = id_to_domain.get(domain_id, f'Domain {domain_id}')
        n_samples_domain = mask.sum()

        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[colors[i]], label=f'{domain_name} (n={n_samples_domain})',
                  alpha=0.7, s=40, edgecolors='black', linewidth=0.5)

    ax.set_title('t-SNE: Learned Feature Representations', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    # Configuration - CLUSTER PATHS
    checkpoint_dir = '/scratch2/macu69pa/hsi_project/checkpoints/pretrain'
    data_root = '/scratch2/macu69pa/hsi_project/data_patches_32x32_split_zarr'
    output_dir = os.path.join(checkpoint_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Domains from training
    domains = ['apfel_vnir', 'arabica', 'coffee', 'grapes', 'paper', 'pottery_full', 'pottery', 'sugar']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find checkpoint
    best_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
    if not os.path.exists(best_checkpoint):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        if len(checkpoints) == 0:
            print("No checkpoints found!")
            return
        best_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    print("\n" + "="*80)
    print("VISUALIZING PRETRAINED MODEL")
    print("="*80)

    # Load checkpoint
    model, history = load_checkpoint(best_checkpoint)

    # Plot training curves
    if len(history['train_losses']) > 0:
        plot_training_curves(
            history,
            os.path.join(output_dir, 'training_curves.png')
        )

    # Create dataloaders
    print("\nCreating dataloaders...")
    print(f"Domains: {domains}")

    train_loader, val_loader, domain_mapping = create_dataloaders(
        data_root=data_root,
        domains=domains,
        batch_size=32,
        num_workers=4,
        train_ratio=0.8
    )
    print(f"  Domain mapping: {domain_mapping}")

    # Visualize
    visualize_reconstructions(
        model, val_loader, device,
        os.path.join(output_dir, 'reconstructions.png'),
        domain_mapping,
        n_samples=16  # More samples to show all 8 domains
    )

    # t-SNE
    analyze_embeddings(
        model, val_loader, device,
        os.path.join(output_dir, 'embeddings_tsne.png'),
        domain_mapping,
        n_samples=400
    )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print(f"Saved to: {output_dir}")
    print("="*80)
    print("Files:")
    print("  1. training_curves.png")
    print("  2. reconstructions.png")
    print("  3. embeddings_tsne.png")
    print("="*80)


if __name__ == '__main__':
    main()
