"""
3D Patch Embedding for Hyperspectral Images
Converts 32×32×256 patches into tokens for transformer
"""

import torch
import torch.nn as nn

class PatchEmbedding3D(nn.Module):
    """
    3D patchification for hyperspectral data.

    Input:  (B, H, W, C) = (B, 32, 32, 256)
    Output: (B, n_spatial, n_spectral, embed_dim)

    Process:
    1. Split spatial: 32×32 → 8×8 patches of 4×4
    2. Split spectral: 256 bands → 16 groups of 16 bands
    3. Embed: 4×4×16 = 256 features → embed_dim
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_h: int = 4,
        patch_w: int = 4,
        patch_c: int = 16,
        in_channels: int = 256,
        embed_dim: int = 128
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_c = patch_c
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.n_spatial_h = img_size // patch_h  # 32 // 4 = 8
        self.n_spatial_w = img_size // patch_w  # 32 // 4 = 8
        self.n_spatial = self.n_spatial_h * self.n_spatial_w  # 64
        self.n_spectral = in_channels // patch_c  # 256 // 16 = 16
        self.n_patches = self.n_spatial * self.n_spectral  # 1024

        # Features per patch
        self.patch_dim = patch_h * patch_w * patch_c  # 4 × 4 × 16 = 256

        # Linear projection?
        self.projection = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) = (B, 32, 32, 256)

        Returns:
            patches: (B, n_spatial, n_spectral, embed_dim)
        """
        B, H, W, C = x.shape

        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}×{W} doesn't match expected {self.img_size}×{self.img_size}"
        assert C == self.in_channels, \
            f"Input channels {C} doesn't match expected {self.in_channels}"

        # Reshape to patches
        # (B, H, W, C) -> (B, n_h, patch_h, n_w, patch_w, n_c, patch_c)
        x = x.reshape(
            B,
            self.n_spatial_h, self.patch_h,
            self.n_spatial_w, self.patch_w,
            self.n_spectral, self.patch_c
        )

        # Rearrange dimensions
        # -> (B, n_h, n_w, n_c, patch_h, patch_w, patch_c)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)

        # Flatten each patch
        # -> (B, n_spatial, n_spectral, patch_dim)
        x = x.reshape(B, self.n_spatial, self.n_spectral, self.patch_dim)

        # Project to embedding dimension
        # -> (B, n_spatial, n_spectral, embed_dim)
        x = self.projection(x)

        return x

    def get_num_patches(self):
        return self.n_spatial, self.n_spectral