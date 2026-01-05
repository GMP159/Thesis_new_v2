"""
SIMPLE Spectral Sanity Check - Uses only model forward pass
FIXED for PyTorch 2.6+
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src1.models.masked_sst import create_model


# def load_model(checkpoint_path):
#     """Load trained model."""
#     # FIXED for PyTorch 2.6+
#     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
#     num_classes = checkpoint['model_state_dict']['classification_head.classifier.weight'].shape[0]
#     model = create_model(num_classes=num_classes)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     return model


def load_model(checkpoint_path):
    """Load trained model - handles DDP checkpoints."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get num_classes (handle DDP prefix)
    key = 'module.classification_head.classifier.weight'
    if key in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict'][key].shape[0]
    else:
        # Fallback: try without 'module.' prefix
        key = 'classification_head.classifier.weight'
        num_classes = checkpoint['model_state_dict'][key].shape[0]
    
    print(f"  Found {num_classes} classes")
    
    # Create model
    model = create_model(num_classes=num_classes)
    
    # Load state dict - remove 'module.' prefix from DDP
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print("  Model loaded successfully!")
    return model




def unpatchify_reconstruction(recon_tokens, H=32, W=32, C=256):
    """Convert reconstruction tokens back to image format."""
    B, n_spatial, n_spectral, patch_dim = recon_tokens.shape

    patch_h, patch_w, patch_c = 4, 4, 16
    n_spatial_h = H // patch_h  # 8
    n_spatial_w = W // patch_w  # 8

    # Reshape: (B, 64, 16, 256) -> (B, 8, 8, 16, 4, 4, 16)
    recon = recon_tokens.reshape(B, n_spatial_h, n_spatial_w, n_spectral, patch_h, patch_w, patch_c)

    # Permute: -> (B, 8, 4, 8, 4, 16, 16)
    recon = recon.permute(0, 1, 4, 2, 5, 3, 6)

    # Reshape to (B, H, W, C)
    recon = recon.reshape(B, H, W, C)

    return recon


def spectral_sanity_check(model, patches, device, bands_to_zero=[64, 128, 192]):
    """
    SIMPLE version: Just zero bands and see what model outputs.
    """
    print("\n" + "="*80)
    print("SPECTRAL SANITY CHECK")
    print("="*80)
    print("Testing if model learned spectral correlations...")
    print(f"Zeroing out bands: {bands_to_zero}")
    print()

    model = model.to(device)
    model.eval()

    B, H, W, C = patches.shape
    original_patches = patches.clone().to(device)

    results = {
        'band_indices': bands_to_zero,
        'reconstruction_errors': [],
        'original_spectra': [],
        'reconstructed_spectra': [],
        'corrupted_spectra': []
    }

    for band_idx in bands_to_zero:
        print(f"Testing band {band_idx}/{C-1}...")

        # Create corrupted version
        corrupted = original_patches.clone()
        corrupted[:, :, :, band_idx] = 0

        with torch.no_grad():
            # Run reconstruction
            recon_tokens, mask = model(corrupted, mode='reconstruction')

            # Unpatchify
            reconstruction = unpatchify_reconstruction(recon_tokens, H, W, C)

        # Get band values
        original_band = original_patches[:, :, :, band_idx].cpu().numpy()
        corrupted_band = corrupted[:, :, :, band_idx].cpu().numpy()
        reconstructed_band = reconstruction[:, :, :, band_idx].cpu().numpy()

        # Compute error on zeroed band
        mse = np.mean((original_band - reconstructed_band) ** 2)
        mae = np.mean(np.abs(original_band - reconstructed_band))

        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")

        results['reconstruction_errors'].append({
            'band': band_idx,
            'mse': mse,
            'mae': mae
        })

        # Store spectra (center pixel)
        center_h, center_w = H // 2, W // 2
        results['original_spectra'].append(original_patches[0, center_h, center_w, :].cpu().numpy())
        results['reconstructed_spectra'].append(reconstruction[0, center_h, center_w, :].cpu().numpy())
        results['corrupted_spectra'].append(corrupted[0, center_h, center_w, :].cpu().numpy())

    return results


def visualize_results(results, save_path):
    """Visualize reconstruction results."""
    print("\nVisualizing results...")

    n_bands = len(results['band_indices'])
    fig, axes = plt.subplots(n_bands, 2, figsize=(14, n_bands * 3))

    if n_bands == 1:
        axes = axes.reshape(1, -1)

    for i, band_idx in enumerate(results['band_indices']):
        original = results['original_spectra'][i]
        corrupted = results['corrupted_spectra'][i]
        reconstructed = results['reconstructed_spectra'][i]
        error_info = results['reconstruction_errors'][i]

        # Full spectrum
        axes[i, 0].plot(original, 'b-', label='Original', linewidth=2, alpha=0.7)
        axes[i, 0].plot(corrupted, 'r--', label='Corrupted', linewidth=1.5, alpha=0.7)
        axes[i, 0].plot(reconstructed, 'g-', label='Reconstructed', linewidth=2, alpha=0.7)
        axes[i, 0].axvline(x=band_idx, color='red', linestyle=':', linewidth=2)
        axes[i, 0].axvspan(band_idx-2, band_idx+2, alpha=0.2, color='red')
        axes[i, 0].set_xlabel('Band Index', fontsize=11)
        axes[i, 0].set_ylabel('Reflectance', fontsize=11)
        axes[i, 0].set_title(f'Band {band_idx} Zeroed - Full Spectrum', fontsize=12, fontweight='bold')
        axes[i, 0].legend(fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)

        # Zoomed view
        zoom_range = 20
        start = max(0, band_idx - zoom_range)
        end = min(len(original), band_idx + zoom_range)
        x_zoom = range(start, end)

        axes[i, 1].plot(x_zoom, original[start:end], 'b-', label='Original', 
                       linewidth=2, marker='o', markersize=3)
        axes[i, 1].plot(x_zoom, corrupted[start:end], 'r--', label='Corrupted', 
                       linewidth=1.5, marker='x', markersize=4)
        axes[i, 1].plot(x_zoom, reconstructed[start:end], 'g-', label='Reconstructed', 
                       linewidth=2, marker='s', markersize=3)
        axes[i, 1].axvline(x=band_idx, color='red', linestyle=':', linewidth=2)
        axes[i, 1].set_xlabel('Band Index', fontsize=11)
        axes[i, 1].set_ylabel('Reflectance', fontsize=11)
        axes[i, 1].set_title(f'Zoomed: Band {band_idx} (MAE={error_info["mae"]:.4f})', 
                            fontsize=12, fontweight='bold')
        axes[i, 1].legend(fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_error_summary(results, save_path):
    """Plot error summary."""
    print("\nCreating error summary...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bands = results['band_indices']
    mse_values = [r['mse'] for r in results['reconstruction_errors']]
    mae_values = [r['mae'] for r in results['reconstruction_errors']]

    axes[0].bar(range(len(bands)), mse_values, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Test Case', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Reconstruction MSE', fontsize=13, fontweight='bold')
    axes[0].set_xticks(range(len(bands)))
    axes[0].set_xticklabels([f'Band {b}' for b in bands])
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mse_values):
        axes[0].text(i, v, f'{v:.5f}', ha='center', va='bottom', fontsize=9)

    axes[1].bar(range(len(bands)), mae_values, color='coral', alpha=0.7)
    axes[1].set_xlabel('Test Case', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Reconstruction MAE', fontsize=13, fontweight='bold')
    axes[1].set_xticks(range(len(bands)))
    axes[1].set_xticklabels([f'Band {b}' for b in bands])
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mae_values):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    checkpoint_path = r'D:\Thesis_new_v2\outputs\checkpoints\pretrain\checkpoint_best.pth'
    data_root = r'D:\Thesis_new\data_patches_32x32_split_zarr'
    output_dir = r'D:\Thesis_new_v2\outputs\checkpoints\pretrain\visualizations'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("SPECTRAL SANITY CHECK")
    print("="*80)
    print(f"Device: {device}")

    print("\nLoading model...")
    model = load_model(checkpoint_path)
    print("  Model loaded")

    print("\nLoading test data...")
    from src1.data.dataset import create_dataloaders

    domains = ['apfel_vnir', 'arabica','coffee', 'grapes', 'paper', 'pottery_full', 'pottery', 'sugar']
    _, val_loader, _ = create_dataloaders(
        data_root=data_root,
        domains=domains,
        batch_size=4,
        num_workers=0,
        train_ratio=0.8
    )

    # Get diverse batch
    for i, (patches, _) in enumerate(val_loader):
        if i == 25:
            break

    print(f"  Test patches: {patches.shape}")

    # Run test
    bands_to_test = [64, 128, 192]
    results = spectral_sanity_check(model, patches, device, bands_to_test)

    # Visualize
    os.makedirs(output_dir, exist_ok=True)
    visualize_results(results, os.path.join(output_dir, 'spectral_sanity_check.png'))
    plot_error_summary(results, os.path.join(output_dir, 'spectral_errors_summary.png'))

    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    for err in results['reconstruction_errors']:
        print(f"  Band {err['band']:3d}: MSE={err['mse']:.6f}, MAE={err['mae']:.4f}")

    avg_mae = np.mean([r['mae'] for r in results['reconstruction_errors']])
    print(f"\nAverage MAE: {avg_mae:.4f}")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    if avg_mae < 0.1:
        print(" [EXCELLENT] Strong spectral correlations learned!")
    elif avg_mae < 0.3:
        print(" [GOOD] Reasonable spectral patterns learned")
    elif avg_mae < 0.5:
        print("  [MODERATE] Some spectral learning")
    else:
        print(" [POOR] Weak spectral learning")
    print("="*80)


if __name__ == '__main__':
    main()