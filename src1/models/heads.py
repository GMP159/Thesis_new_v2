"""
Output heads for pre-training and fine-tuning
"""

import torch
import torch.nn as nn

class ReconstructionHead(nn.Module):
    """
    Reconstruction head for masked pre-training.
    Projects embeddings back to patch features for reconstruction.
    """

    def __init__(self, embed_dim: int = 128, patch_dim: int = 256):
        super().__init__()
        self.decoder = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        """
        Args:
            x: (B, n_spatial, n_spectral, embed_dim)

        Returns:
            reconstruction: (B, n_spatial, n_spectral, patch_dim)
        """
        return self.decoder(x)


class ClassificationHead(nn.Module):
    """
    Classification head for fine-tuning.
    Global average pooling + linear classifier.
    """

    def __init__(self, embed_dim: int = 128, num_classes: int = 9):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, n_spatial, n_spectral, embed_dim)

        Returns:
            logits: (B, num_classes)
        """
        # Global average pooling over all tokens
        x_pooled = x.mean(dim=[1, 2])  # (B, embed_dim)
        logits = self.classifier(x_pooled)
        return logits
