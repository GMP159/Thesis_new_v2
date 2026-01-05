"""
Dataset for loading HSI patches from zarr files
IMPROVED VERSION with better handling
"""

import torch
import zarr
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
import os
import glob
import logging
from src1.data.transforms import Compose

logger = logging.getLogger(__name__)


class MultiDomainHSIDataset(Dataset):
    """
    Multi-domain HSI dataset loading from zarr files.

    Your data structure:
    - Already at 256 bands  
    - Already cleaned (no invalid patches)  
    - Stored as zarr files per domain (some split into parts)

    Args:
        zarr_paths: List of paths to zarr files
        split: 'train' or 'val'
        train_ratio: Ratio of train split (0.8 = 80% train, 20% val)
        return_domain_labels: If True, returns domain ID as label
        normalize_per_patch: If True, normalize each patch independently
    """

    def __init__(
        self,
        zarr_paths: List[str],
        split: str = 'train',
        train_ratio: float = 0.8,
        return_domain_labels: bool = True,
        normalize_per_patch: bool = True,
        seed: int = 42,
        transform = None
    ):
        self.zarr_paths = zarr_paths
        self.split = split
        self.return_domain_labels = return_domain_labels
        self.normalize_per_patch = normalize_per_patch
        self.transform = transform
        
        np.random.seed(seed)  # For reproducible splits

        # Build index of all patches
        self.samples = []
        self.domain_to_id = {}
        self.domain_counts = {}

        logger.info(f"Loading {split} split from {len(zarr_paths)} zarr files...")

        domain_id = 0
        for zarr_path in zarr_paths:
            try:
                # Open zarr - try format 2 first, then default
                try:
                    z = zarr.open(zarr_path, mode='r', zarr_version=2)
                except Exception:
                    z = zarr.open(zarr_path, mode='r')

                # Get domain info - with fallback to extracting from path
                try:
                    domain_name_raw = z['domain_name'][()]
                    if isinstance(domain_name_raw, bytes):
                        domain_name = domain_name_raw.decode('utf-8')
                    else:
                        domain_name = str(domain_name_raw)
                except (KeyError, Exception):
                    # Extract domain name from file path if metadata not available
                    basename = os.path.basename(zarr_path).replace('_patches.zarr', '').replace('.zarr', '')
                    domain_name = basename
                    logger.warning(f"  Could not read domain_name from {zarr_path}, extracting from path: {domain_name}")

                # Assign domain ID (consistent across all parts of same domain)
                if domain_name not in self.domain_to_id:
                    self.domain_to_id[domain_name] = domain_id
                    self.domain_counts[domain_name] = 0
                    domain_id += 1

                # Get patches info
                n_patches = z['patches'].shape[0]
                try:
                    num_bands = z['num_bands'][()]
                except (KeyError, Exception):
                    # Infer from patch shape if num_bands not available
                    num_bands = z['patches'].shape[-1]
                    logger.warning(f"  Could not read num_bands from {zarr_path}, inferring from patches: {num_bands}")

                # Verify data shape
                patch_shape = z['patches'].shape
                assert patch_shape[1:] == (32, 32, 256), \
                    f"Unexpected patch shape: {patch_shape}. Expected (N, 32, 32, 256)"

                # Split into train/val using indices
                indices = np.arange(n_patches)
                np.random.shuffle(indices)  # Shuffle for random split
                
                n_train = int(n_patches * train_ratio)
                if split == 'train':
                    split_indices = indices[:n_train]
                else:
                    split_indices = indices[n_train:]

                logger.info(f"  {domain_name} ({os.path.basename(zarr_path)}): "
                          f"{len(split_indices)} patches (total={n_patches}, bands={num_bands})")

                # Add to samples list
                for idx in split_indices:
                    self.samples.append({
                        'zarr_path': zarr_path,
                        'patch_idx': int(idx),  # Important: convert to int!
                        'domain_id': self.domain_to_id[domain_name],
                        'domain_name': domain_name
                    })
                    self.domain_counts[domain_name] += 1

            except Exception as e:
                logger.error(f"Error loading {zarr_path}: {e}")
                continue

        logger.info(f"Total {split} samples: {len(self.samples)}")
        logger.info(f"Domain mapping: {self.domain_to_id}")
        logger.info(f"Domain counts: {self.domain_counts}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        try:
            # Load patch from zarr
            z = zarr.open(sample['zarr_path'], mode='r')
            patch = z['patches'][sample['patch_idx']]  # (32, 32, 256)

            # Convert to float32
            patch = patch.astype(np.float32)

            # Normalize
            if self.normalize_per_patch:
                # Per-patch normalization (better for pre-training)
                patch_mean = patch.mean()
                patch_std = patch.std()
                if patch_std < 1e-6:
                    patch_std = 1.0
                patch = (patch - patch_mean) / patch_std
            else:
                # Per-band normalization (better for fine-tuning)
                patch_mean = patch.mean(axis=(0, 1), keepdims=True)
                patch_std = patch.std(axis=(0, 1), keepdims=True)
                patch_std = np.where(patch_std < 1e-6, 1.0, patch_std)
                patch = (patch - patch_mean) / patch_std
            
            # Apply transforms if any
            if self.transform is not None:
                patch = self.transform(patch)

            # Convert to tensor (keep as H, W, C for model input)
            patch = torch.from_numpy(patch).float()

            if self.return_domain_labels:
                label = sample['domain_id']
                return patch, label
            else:
                return patch
                
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(np.random.randint(0, len(self)))

    def get_domain_info(self) -> Dict:
        """Get information about domains."""
        return {
            'num_domains': len(self.domain_to_id),
            'domain_to_id': self.domain_to_id,
            'domain_counts': self.domain_counts,
            'total_samples': len(self.samples)
        }


def find_zarr_files(data_root: str, domains: List[str]) -> List[str]:
    """
    Find all zarr files for given domains, handling split files.
    
    Args:
        data_root: Root directory with zarr files
        domains: List of domain names
        
    Returns:
        List of zarr file paths
    """
    zarr_paths = []
    
    for domain in domains:
        # Try common patterns
        patterns = [
            os.path.join(data_root, f"{domain}_patches.zarr"),
            os.path.join(data_root, f"{domain}.zarr"),
        ]
        
        found = False
        for pattern in patterns:
            if os.path.exists(pattern):
                zarr_paths.append(pattern)
                found = True
                break
        
        if not found:
            # Try glob for split files (part1_of_N, etc.)
            matches = sorted(glob.glob(os.path.join(data_root, f"{domain}_part*.zarr")))
            if matches:
                zarr_paths.extend(matches)
                logger.info(f"Found {len(matches)} parts for domain '{domain}'")
            else:
                # Try alternative patterns
                matches = sorted(glob.glob(os.path.join(data_root, f"{domain}*.zarr")))
                if matches:
                    zarr_paths.extend(matches)
                    logger.info(f"Found {len(matches)} files for domain '{domain}'")
                else:
                    logger.warning(f"No zarr files found for domain '{domain}'")
    
    return zarr_paths


def create_dataloaders(
    data_root: str,
    domains: List[str],
    batch_size: int = 128,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    train_transforms=None,
    val_transforms=None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Create train and validation dataloaders.

    Args:
        data_root: Root directory containing zarr files
        domains: List of domain names to include
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Train/val split ratio
        seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, domain_mapping
    """
    # Find zarr files
    zarr_paths = find_zarr_files(data_root, domains)

    if len(zarr_paths) == 0:
        raise ValueError(f"No zarr files found for domains: {domains}")

    logger.info(f"Found {len(zarr_paths)} zarr files total")

    # Create datasets
    train_dataset = MultiDomainHSIDataset(
        zarr_paths=zarr_paths,
        split='train',
        train_ratio=train_ratio,
        return_domain_labels=True,
        normalize_per_patch=True,  # Per-patch for pre-training
        seed=seed,
        transform=train_transforms
    )

    val_dataset = MultiDomainHSIDataset(
        zarr_paths=zarr_paths,
        split='val',
        train_ratio=train_ratio,
        return_domain_labels=True,
        normalize_per_patch=True,
        seed=seed,
        transform=val_transforms
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,  # Drop last incomplete batch
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, train_dataset.domain_to_id


if __name__ == "__main__":
    # Test dataset
    logging.basicConfig(level=logging.INFO)
    
    data_root = r"D:\Thesis_new\data_patches_32x32_split_zarr"
    domains = ['branches']
    
    print("Testing dataset creation...")
    train_loader, val_loader, domain_mapping = create_dataloaders(
        data_root=data_root,
        domains=domains,
        batch_size=4,
        num_workers=0,
        train_ratio=0.8
    )
    
    print(f"\nDomain mapping: {domain_mapping}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading a batch
    print("\nTesting batch loading...")
    patches, labels = next(iter(train_loader))
    print(f"Patches shape: {patches.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Patches dtype: {patches.dtype}")
    print(f"Patches range: [{patches.min():.3f}, {patches.max():.3f}]")
    print(f"Labels: {labels}")
    print("\n  Dataset test passed!")