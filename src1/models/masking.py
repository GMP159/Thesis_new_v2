"""
Random masking for self-supervised pre-training
"""

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class TubeMasking(nn.Module):
    def __init__(self, mask_ratio: float = 0.85):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x, mask_token):
        # x shape: (B, n_spatial, n_spectral, D) -> (B, 64, 16, 128)
        B, n_spatial, n_spectral, D = x.shape
        
        # 1. Decide which pixels to mask (B, 64)
        noise = torch.rand(B, n_spatial, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # 2. Create spatial mask
        num_masked = int(n_spatial * self.mask_ratio)
        mask_spatial = torch.zeros(B, n_spatial, device=x.device)
        mask_spatial[:, :num_masked] = 1 # 1 = masked
        
        # 3. Unshuffle to original order
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_spatial = torch.gather(mask_spatial, dim=1, index=ids_restore)
        
        # 4. Expand to "Tubes": (B, 64) -> (B, 64, 1) -> (B, 64, 16)
        mask = mask_spatial.unsqueeze(-1).repeat(1, 1, n_spectral)
        
        # 5. Apply to tokens
        mask_expanded = mask.unsqueeze(-1) # (B, 64, 16, 1)
        x_masked = x * (1 - mask_expanded) + mask_token * mask_expanded
        
        return x_masked, mask

class RandomMasking(nn.Module):
    """
    Random masking of tokens for masked reconstruction pre-training.

    Args:
        mask_ratio: Ratio of tokens to mask (0.75 = 75%)
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x, mask_token):
        """
        Args:
            x: (B, n_spatial, n_spectral, embed_dim)
            mask_token: (1, 1, 1, embed_dim) learnable mask token

        Returns:
            x_masked: (B, n_spatial, n_spectral, embed_dim) with masked tokens
            mask: (B, n_spatial, n_spectral) binary mask (1=masked, 0=visible)
        """
        B, n_spatial, n_spectral, embed_dim = x.shape
        N = n_spatial * n_spectral  # Total tokens

        # Flatten for easier masking
        x_flat = x.reshape(B, N, embed_dim)  # (B, 1024, 128)

        # Generate random mask for each sample in batch
        n_masked = int(N * self.mask_ratio)
        mask = torch.zeros(B, N, device=x.device)  # (B, 1024)

        for i in range(B):
            # Random permutation
            perm = torch.randperm(N, device=x.device)
            # Mask the first n_masked tokens
            mask[i, perm[:n_masked]] = 1

        # Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1)  # (B, 1024, 1)

        # Replace masked tokens with mask token
        mask_token_expanded = mask_token.expand(B, N, embed_dim)
        x_masked_flat = x_flat * (1 - mask_expanded) + mask_token_expanded * mask_expanded

        # Reshape back
        x_masked = x_masked_flat.reshape(B, n_spatial, n_spectral, embed_dim)
        mask_reshaped = mask.reshape(B, n_spatial, n_spectral)

        return x_masked, mask_reshaped
