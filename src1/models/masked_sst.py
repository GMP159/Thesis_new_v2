"""
Masked Spatial-Spectral Transformer for Hyperspectral Image Classification
Main model class combining all components
"""

import torch
import torch.nn as nn
from src1.models.patch_embedding import PatchEmbedding3D
from src1.models.positional_encoding import PositionalEncoding3D
from src1.models.transformer_block import FactorizedSSTransformerBlock
# from src1.models.masking import RandomMasking
from src1.models.masking import TubeMasking
from src1.models.heads import ReconstructionHead, ClassificationHead

class MaskedSST(nn.Module):
    """
    Masked Spatial-Spectral Transformer.

    Architecture:
    1. 3D Patch Embedding: (B, 32, 32, 256) -> (B, 64, 16, 128)
    2. Positional Encoding: Add 3D position information
    3. [Optional] Masking: Random 75% for pre-training
    4. Transformer Blocks: Factorized attention (×4)
    5. Output Head: Reconstruction or Classification

    Args:
        img_size: Input image size (32)
        patch_h: Spatial patch height (4)
        patch_w: Spatial patch width (4)
        patch_c: Spectral patch channels (16)
        in_channels: Input spectral bands (256)
        embed_dim: Embedding dimension (128)
        depth: Number of transformer blocks (4)
        num_heads: Number of attention heads (8)
        mlp_ratio: MLP hidden dim ratio (4)
        dropout: Dropout rate (0.1)
        num_classes: Number of classes for classification (9)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_h: int = 4,
        patch_w: int = 4,
        patch_c: int = 16,
        in_channels: int = 256,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        num_classes: int = 9
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            patch_h=patch_h,
            patch_w=patch_w,
            patch_c=patch_c,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        n_spatial, n_spectral = self.patch_embed.get_num_patches()
        self.n_spatial = n_spatial  # 64
        self.n_spectral = n_spectral  # 16

        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(
            n_spatial_h=int(n_spatial ** 0.5),
            n_spatial_w=int(n_spatial ** 0.5),
            n_spectral=n_spectral,
            embed_dim=embed_dim
        )

        # Mask token (learnable)
        # self.mask_token = nn.Parameter(torch.randn(1, 1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Masking module
        # self.masking = RandomMasking(mask_ratio=0.75)

        self.masking = TubeMasking(mask_ratio=0.90)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            FactorizedSSTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                n_spatial=n_spatial,
                n_spectral=n_spectral,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Output heads
        patch_dim = patch_h * patch_w * patch_c  # 256
        self.reconstruction_head = ReconstructionHead(embed_dim, patch_dim)
        self.classification_head = ClassificationHead(embed_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, apply_masking=False):
        """
        Forward pass through encoder.

        Args:
            x: (B, H, W, C) = (B, 32, 32, 256)
            apply_masking: Whether to apply masking (True for pre-training)

        Returns:
            features: (B, n_spatial, n_spectral, embed_dim)
            mask: (B, n_spatial, n_spectral) if apply_masking else None
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, 64, 16, 128)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply masking if needed
        mask = None
        if apply_masking:
            x, mask = self.masking(x, self.mask_token)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x, mask

    def forward_reconstruction(self, x, apply_masking=True):
        """
        Forward pass for pre-training (reconstruction).

        Args:
            x: (B, 32, 32, 256)

        Returns:
            reconstruction: (B, n_spatial, n_spectral, patch_dim)
            mask: (B, n_spatial, n_spectral)
        """
        features, mask = self.forward_encoder(x, apply_masking=apply_masking)
        reconstruction = self.reconstruction_head(features)
        return reconstruction, mask

    def forward_classification(self, x):
        """
        Forward pass for fine-tuning (classification).

        Args:
            x: (B, 32, 32, 256)

        Returns:
            logits: (B, num_classes)
        """
        features, _ = self.forward_encoder(x, apply_masking=False)
        logits = self.classification_head(features)
        return logits

    def forward(self, x, mode='classification'):
        """
        Forward pass. Mode determines which head to use.

        Args:
            x: (B, 32, 32, 256)
            mode: 'reconstruction' or 'classification'

        Returns:
            Output depends on mode
        """
        if mode == 'reconstruction':
            return self.forward_reconstruction(x)
        else:
            return self.forward_classification(x)


def create_model(num_classes=9):
    """Create model with default hyperparameters."""
    model = MaskedSST(
        img_size=32,
        patch_h=4,
        patch_w=4,
        patch_c=16,
        in_channels=256,
        embed_dim=128,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_classes=num_classes
    )
    return model


if __name__ == '__main__':
    # Test model
    model = create_model(num_classes=9)

    # Test input
    x = torch.randn(2, 32, 32, 256)

    # Test reconstruction
    recon, mask = model(x, mode='reconstruction')
    print(f"Reconstruction output: {recon.shape}")  # (2, 64, 16, 256)
    print(f"Mask shape: {mask.shape}")  # (2, 64, 16)
    print(f"Masked ratio: {mask.float().mean():.2f}")  # ~0.75

    # Test classification
    logits = model(x, mode='classification')
    print(f"Classification output: {logits.shape}")  # (2, 9)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
