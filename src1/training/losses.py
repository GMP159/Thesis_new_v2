"""
Loss functions for pre-training and fine-tuning
ACTUALLY FIXED VERSION - Correct broadcasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def patchify_target(x, patch_h=4, patch_w=4, patch_c=16):
    """
    Patchify input to match model's patch embedding.
    Must match EXACTLY how PatchEmbedding3D works!

    Args:
        x: (B, H, W, C) = (B, 32, 32, 256)
        patch_h: 4
        patch_w: 4
        patch_c: 16

    Returns:
        patches: (B, n_spatial, n_spectral, patch_dim)
                 (B, 64, 16, 256)
    """
    B, H, W, C = x.shape

    n_spatial_h = H // patch_h  # 8
    n_spatial_w = W // patch_w  # 8
    n_spectral = C // patch_c  # 16
    patch_dim = patch_h * patch_w * patch_c  # 256

    # Reshape to patches - SAME as PatchEmbedding3D
    x = x.reshape(
        B,
        n_spatial_h, patch_h,
        n_spatial_w, patch_w,
        n_spectral, patch_c
    )

    # Rearrange dimensions
    x = x.permute(0, 1, 3, 5, 2, 4, 6)

    # Flatten each patch
    n_spatial = n_spatial_h * n_spatial_w
    x = x.reshape(B, n_spatial, n_spectral, patch_dim)

    return x

def masked_l1_loss(pred, target, mask):
    """
    L1 loss on masked tokens only.
    ACTUALLY FIXED: Properly account for patch_dim in normalization!

    Args:
        pred: (B, n_spatial, n_spectral, patch_dim)
        target: (B, n_spatial, n_spectral, patch_dim)
        mask: (B, n_spatial, n_spectral) binary, 1=masked, 0=visible

    Returns:
        loss: Scalar loss
    """
    B, n_spatial, n_spectral, patch_dim = pred.shape

    # Expand mask for all features
    mask_expanded = mask.unsqueeze(-1)  # (B, n_spatial, n_spectral, 1)

    # Compute L1 loss per element
    loss = torch.abs(pred - target)  # (B, n_spatial, n_spectral, patch_dim)

    # Apply mask (broadcasting happens here!)
    masked_loss = loss * mask_expanded  # (B, n_spatial, n_spectral, patch_dim)

    # THE FIX: Count masked elements correctly
    # mask has shape (B, n_spatial, n_spectral)
    # Each masked token contributes patch_dim elements
    n_masked_tokens = mask.sum()  # Number of masked tokens
    n_masked_elements = n_masked_tokens * patch_dim  # Total masked elements

    if n_masked_tokens == 0:
        return torch.tensor(0.0, device=pred.device)

    # Sum all losses and divide by total masked elements
    total_loss = masked_loss.sum()
    avg_loss = total_loss / n_masked_elements

    return avg_loss


def reconstruction_loss(model_output, original_patches, mask):
    """
    Complete reconstruction loss for pre-training.

    Args:
        model_output: (B, n_spatial, n_spectral, patch_dim) from decoder
        original_patches: (B, 32, 32, 256) original input
        mask: (B, n_spatial, n_spectral) binary mask

    Returns:
        loss: Scalar loss
    """
    # Patchify original input to match model output
    target_patches = patchify_target(
        original_patches,
        patch_h=4,
        patch_w=4,
        patch_c=16
    )

    # Compute masked L1 loss
    loss = masked_l1_loss(model_output, target_patches, mask)

    return loss


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for classification with label smoothing."""

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, num_classes)
            labels: (B,) class indices

        Returns:
            loss: Scalar loss
        """
        return self.ce_loss(logits, labels)


def compute_reconstruction_accuracy(pred, target, mask, threshold=1.0):
    """
    Compute "accuracy" of reconstruction (% of features within threshold).
    ACTUALLY FIXED: Properly account for patch_dim!

    Args:
        pred: (B, n_spatial, n_spectral, patch_dim)
        target: (B, n_spatial, n_spectral, patch_dim)
        mask: (B, n_spatial, n_spectral)
        threshold: Consider "correct" if |pred - target| < threshold

    Returns:
        accuracy: Scalar in [0, 100] (percentage)
    """
    B, n_spatial, n_spectral, patch_dim = pred.shape

    mask_expanded = mask.unsqueeze(-1)  # (B, n_spatial, n_spectral, 1)

    # Check if within threshold
    diff = torch.abs(pred - target)
    correct = (diff < threshold).float()  # (B, n_spatial, n_spectral, patch_dim)

    # Apply mask (broadcasting!)
    masked_correct = correct * mask_expanded

    # THE FIX: Count correctly with patch_dim
    n_correct = masked_correct.sum()  # Total correct elements
    n_masked_tokens = mask.sum()  # Number of masked tokens
    n_total_elements = n_masked_tokens * patch_dim  # Total masked elements

    if n_masked_tokens == 0:
        return 0.0

    # Return as percentage
    accuracy = 100.0 * n_correct / n_total_elements

    return accuracy.item()


if __name__ == "__main__":
    # Test with known values
    print("Testing ACTUALLY FIXED loss functions...")
    print("="*80)

    # Create test data
    B, n_spatial, n_spectral, patch_dim = 2, 64, 16, 256

    # Prediction and target that are close
    pred = torch.randn(B, n_spatial, n_spectral, patch_dim) * 0.5
    target = pred + 0.2 * torch.randn_like(pred)  # Small noise

    # Create mask (75% masked)
    mask = torch.rand(B, n_spatial, n_spectral) < 0.75
    mask = mask.float()

    print(f"Test setup:")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask ratio: {mask.mean():.2%}")

    # Calculate expected values manually
    n_masked_tokens = int(mask.sum().item())
    n_masked_elements = n_masked_tokens * patch_dim
    print(f"  Masked tokens: {n_masked_tokens}")
    print(f"  Masked elements: {n_masked_elements}")
    print()

    # Test loss
    loss = masked_l1_loss(pred, target, mask)
    print(f"Masked L1 loss: {loss.item():.4f}")
    print(f"  Expected range: 0.1-0.3 (small noise added)")
    print(f"    PASS" if 0.05 < loss.item() < 0.5 else "    FAIL")
    print()

    # Test accuracy
    acc = compute_reconstruction_accuracy(pred, target, mask, threshold=1.0)
    print(f"Reconstruction accuracy: {acc:.2f}%")
    print(f"  Expected range: 60-95% (small noise)")
    print(f"    PASS" if 50 < acc < 100 else "    FAIL")
    print()

    # Test with original patches
    print("Testing reconstruction_loss...")
    original = torch.randn(2, 32, 32, 256) * 0.5
    model_output = torch.randn(2, 64, 16, 256) * 0.5
    mask_2d = torch.rand(2, 64, 16) < 0.75
    mask_2d = mask_2d.float()

    loss = reconstruction_loss(model_output, original, mask_2d)
    print(f"Reconstruction loss: {loss.item():.4f}")
    print(f"  Expected range: 0.3-0.8")
    print(f"    PASS" if 0.2 < loss.item() < 1.0 else "    FAIL")
    print()

    print("="*80)
    if 0.05 < loss.item() < 1.0 and 50 < acc < 100:
        print("  ALL TESTS PASSED! Loss values are correct.")
    else:
        print("  TESTS FAILED! Something is still wrong.")