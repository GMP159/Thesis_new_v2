"""
Data augmentation for HSI patches
"""

import torch
import numpy as np

class RandomFlip:
    """Random horizontal and vertical flips."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array or tensor

        Returns:
            patch: Flipped patch
        """
        if np.random.random() < self.p:
            # Horizontal flip
            patch = np.flip(patch, axis=1).copy()

        if np.random.random() < self.p:
            # Vertical flip
            patch = np.flip(patch, axis=0).copy()

        return patch


class RandomRotate90:
    """Random 90-degree rotations."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array

        Returns:
            patch: Rotated patch
        """
        if np.random.random() < self.p:
            # Random 90-degree rotation
            k = np.random.randint(1, 4)  # 1, 2, or 3 times 90°
            patch = np.rot90(patch, k=k, axes=(0, 1)).copy()

        return patch


class SpectralNoise:
    """Add Gaussian noise to spectral dimension."""

    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array

        Returns:
            patch: Patch with added noise
        """
        noise = np.random.normal(0, self.noise_std, patch.shape)
        patch = patch + noise.astype(patch.dtype)
        return patch


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, patch):
        for t in self.transforms:
            patch = t(patch)
        return patch


# Pre-defined augmentation pipeline
def get_train_transforms():
    """Get standard training augmentations."""
    return Compose([
        RandomFlip(p=0.5),
        RandomRotate90(p=0.5),
        SpectralNoise(noise_std=0.01)
    ])


def get_val_transforms():
    """No augmentation for validation."""
    return None
