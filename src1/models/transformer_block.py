"""
Factorized Spatial-Spectral Transformer Block
"""

import torch
import torch.nn as nn

class FactorizedSSTransformerBlock(nn.Module):
    """
    Factorized Spatial-Spectral Transformer Block.

    Performs attention in factorized manner:
    1. Spatial attention (across spatial positions for each spectral group)
    2. Spectral attention (across spectral groups for each spatial position)

    This reduces complexity from O(N²) to O(n_spatial² + n_spectral²)
    For 64 spatial × 16 spectral = 1024 tokens:
    - Standard: O(1024²) = 1M operations
    - Factorized: O(64² + 16²) = 4K operations (250× faster!)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        n_spatial: int = 64,
        n_spectral: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_spatial = n_spatial
        self.n_spectral = n_spectral

        # Layer norms
        self.norm1_spatial = nn.LayerNorm(embed_dim)
        self.norm2_spatial = nn.LayerNorm(embed_dim)
        self.norm1_spectral = nn.LayerNorm(embed_dim)
        self.norm2_spectral = nn.LayerNorm(embed_dim)

        # Spatial attention
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Spectral attention
        self.spectral_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward networks
        mlp_hidden_dim = embed_dim * mlp_ratio
        self.mlp_spatial = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.mlp_spectral = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    # In FactorizedSSTransformerBlock.forward()

    def forward(self, x):
        B = x.shape[0]

        # === SPATIAL ATTENTION ===
        # Process each spectral group independently
        # Current: (B, n_spatial, n_spectral, D)
        # Target:  (B*n_spectral, n_spatial, D)
        
        # FIX: Permute first to bring Spectral next to Batch
        x_perm = x.permute(0, 2, 1, 3) # (B, n_spectral, n_spatial, D)
        x_spatial = x_perm.reshape(B * self.n_spectral, self.n_spatial, self.embed_dim)

        # ... (Perform Attention & MLP) ...
        x_norm = self.norm1_spatial(x_spatial)
        attn_out, _ = self.spatial_attn(x_norm, x_norm, x_norm)
        x_spatial = x_spatial + attn_out
        
        x_norm = self.norm2_spatial(x_spatial)
        mlp_out = self.mlp_spatial(x_norm)
        x_spatial = x_spatial + mlp_out

        # FIX: Reshape back correctly
        x_spatial = x_spatial.reshape(B, self.n_spectral, self.n_spatial, self.embed_dim)
        x = x_spatial.permute(0, 2, 1, 3) # (B, n_spatial, n_spectral, D)


        # === SPECTRAL ATTENTION ===
        # Process each spatial position independently
        # Current: (B, n_spatial, n_spectral, D)
        # Target:  (B*n_spatial, n_spectral, D)
        
        # This part was actually OK logic-wise, but remove the useless permute(0,1,2,3)
        x_spectral = x.reshape(B * self.n_spatial, self.n_spectral, self.embed_dim)

        # ... (Perform Attention & MLP) ...
        x_norm = self.norm1_spectral(x_spectral)
        attn_out, _ = self.spectral_attn(x_norm, x_norm, x_norm)
        x_spectral = x_spectral + attn_out
        
        x_norm = self.norm2_spectral(x_spectral)
        mlp_out = self.mlp_spectral(x_norm)
        x_spectral = x_spectral + mlp_out

        # Reshape back
        x = x_spectral.reshape(B, self.n_spatial, self.n_spectral, self.embed_dim)

        return x
