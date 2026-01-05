"""
Main script for pre-training MaskedSST model
ENHANCED with W&B support
"""

import torch
import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src1.models.masked_sst import create_model
from src1.data.dataset import create_dataloaders
from src1.training.pretrain_trainer import PretrainTrainer  # Will use enhanced version
from src1.data.transforms import get_train_transforms, get_val_transforms


def setup_logging(save_dir):
    """Setup logging to file and console."""
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )


def main(args):
    # Setup logging
    setup_logging(args.save_dir)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("MASKED SPATIAL-SPECTRAL TRANSFORMER - PRE-TRAINING")
    logger.info("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Memory: {memory_gb:.2f} GB")

    # Domains
    if args.all_domains:
        domains = [
            'apfel_swir', 'apfel_vnir', 'arabica', 'branches',
            'coffee', 'grapes', 'paper', 'pottery_full', 'pottery', 'sugar'
        ]
    else:
        domains = args.domains

    logger.info(f"Training on {len(domains)} domains: {domains}")

    # Create transforms
    logger.info("Creating data transforms...")
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()


    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, domain_mapping = create_dataloaders(
            data_root=args.data_root,
            domains=domains,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            train_transforms=train_transforms,
            val_transforms=val_transforms
        )
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        raise

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Domain mapping: {domain_mapping}")

    # Create model
    logger.info("Creating model...")
    num_classes = len(domain_mapping)
    model = create_model(num_classes=num_classes)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")
    logger.info(f"Model size: {n_params * 4 / 1e6:.2f} MB (float32)")

    # Create trainer with gradient monitoring and W&B
    logger.info("Creating trainer...")
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        save_dir=args.save_dir,
        warmup_epochs=args.warmup_epochs,
        save_freq=args.save_freq,
        # NEW: W&B and gradient monitoring
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_gradients_every=args.log_gradients_every
    )

    # Log hyperparameters
    logger.info("\nHyperparameters:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Warmup epochs: {args.warmup_epochs}")
    logger.info(f"  Gradient logging: Every {args.log_gradients_every} batches")
    logger.info(f"  W&B enabled: {args.use_wandb}")
    if args.use_wandb:
        logger.info(f"  W&B project: {args.wandb_project}")
        logger.info(f"  W&B run name: {args.wandb_run_name or 'auto-generated'}")

    # Start training
    logger.info("\nStarting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        logger.info("Checkpoint saved")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-train MaskedSST with gradient monitoring and W&B',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument('--data_root', type=str,
                       default=r'D:\Thesis_new\data_patches_32x32_split_zarr',
                       help='Root directory with zarr files')
    parser.add_argument('--domains', nargs='+',
                       default=['branches','apfel_swir','grapes','pottery'],
                       help='List of domains to use')
    parser.add_argument('--all_domains', action='store_true',
                       help='Use all available domains')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train/val split ratio')

    # Training
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')

    # Logging and checkpointing
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str,
                       default='outputs/checkpoints/pretrain',
                       help='Directory to save checkpoints')

    # NEW: Gradient monitoring and W&B
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str,
                       default='hsi-pretraining',
                       help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name (auto-generated if not specified)')
    parser.add_argument('--log_gradients_every', type=int, default=5,
                       help='Log gradient statistics every N batches')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    # Validate arguments
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if not os.path.exists(args.data_root):
        raise ValueError(f"Data root does not exist: {args.data_root}")

    main(args)