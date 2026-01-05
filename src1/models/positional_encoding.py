"""
3D Positional Encoding for Spatial-Spectral Tokens
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding3D(nn.Module):
    """
    3D sinusoidal positional encoding for (h, w, c) positions.

    Args:
        n_spatial_h: Number of spatial patches in height (8)
        n_spatial_w: Number of spatial patches in width (8)
        n_spectral: Number of spectral groups (16)
        embed_dim: Embedding dimension (128)
    """

    def __init__(
        self,
        n_spatial_h: int = 8,
        n_spatial_w: int = 8,
        n_spectral: int = 16,
        embed_dim: int = 128
    ):
        super().__init__()

        self.n_spatial_h = n_spatial_h
        self.n_spatial_w = n_spatial_w
        self.n_spectral = n_spectral
        self.embed_dim = embed_dim

        # Split embedding dimensions for each axis
        self.dim_h = embed_dim // 3
        self.dim_w = embed_dim // 3
        self.dim_c = embed_dim - self.dim_h - self.dim_w  # Remaining dims

        # Create positional encoding
        pos_encoding = self._create_positional_encoding()
        self.register_buffer('pos_encoding', pos_encoding)

    def _create_positional_encoding(self):
        """Create 3D sinusoidal positional encoding."""
        n_spatial = self.n_spatial_h * self.n_spatial_w
        pos_encoding = torch.zeros(n_spatial, self.n_spectral, self.embed_dim)

        # Create encodings for each dimension
        for h in range(self.n_spatial_h):
            for w in range(self.n_spatial_w):
                spatial_idx = h * self.n_spatial_w + w

                for c in range(self.n_spectral):
                    # Height encoding
                    pos_encoding[spatial_idx, c, :self.dim_h] = \
                        self._get_encoding(h, self.n_spatial_h, self.dim_h)

                    # Width encoding
                    pos_encoding[spatial_idx, c, self.dim_h:self.dim_h+self.dim_w] = \
                        self._get_encoding(w, self.n_spatial_w, self.dim_w)

                    # Spectral encoding
                    pos_encoding[spatial_idx, c, self.dim_h+self.dim_w:] = \
                        self._get_encoding(c, self.n_spectral, self.dim_c)

        return pos_encoding

    def _get_encoding(self, pos, max_pos, dim):
        """Get sinusoidal encoding for a single position."""
        encoding = torch.zeros(dim)

        for i in range(0, dim, 2):
            denominator = math.pow(10000, 2 * i / dim)
            encoding[i] = math.sin(pos / denominator)
            if i + 1 < dim:
                encoding[i + 1] = math.cos(pos / denominator)

        return encoding

    def forward(self, x):
        """
        Args:
            x: (B, n_spatial, n_spectral, embed_dim)

        Returns:
            x + pos_encoding: (B, n_spatial, n_spectral, embed_dim)
        """
        return x + self.pos_encoding.unsqueeze(0)
