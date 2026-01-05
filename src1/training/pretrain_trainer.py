"""
Trainer for masked reconstruction pre-training
ENHANCED with gradient monitoring and Weights & Biases
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import logging
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from src1.training.losses import reconstruction_loss, compute_reconstruction_accuracy

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class GradientMonitor:
    """Monitor gradient statistics."""

    def compute_stats(self, model):
        """Compute gradient statistics."""
        stats = {}
        all_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                all_norms.append(grad_norm)

                # Store per-layer (optional, can be verbose)
                stats[name] = grad_norm

        # Summary statistics
        if all_norms:
            return {
                'grad/norm_mean': np.mean(all_norms),
                'grad/norm_max': np.max(all_norms),
                'grad/norm_min': np.min(all_norms),
                'grad/norm_std': np.std(all_norms)
            }
        return {}


class PretrainTrainer:
    """
    Trainer for self-supervised pre-training via masked reconstruction.
    ENHANCED with gradient monitoring and W&B support.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device='cuda',
        lr=1e-3,
        weight_decay=0.05,
        epochs=200,
        save_dir='checkpoints/pretrain',
        warmup_epochs=10,
        save_freq=10,
        # NEW: Gradient monitoring and W&B
        use_wandb=False,
        wandb_project='hsi-pretraining',
        wandb_run_name=None,
        log_gradients_every=10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.warmup_epochs = warmup_epochs
        self.save_freq = save_freq

        # NEW: Gradient monitoring
        self.log_gradients_every = log_gradients_every
        self.grad_monitor = GradientMonitor()
        self.gradient_norms = []  # Store per epoch

        # NEW: Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb not installed. Install with: pip install wandb")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        'lr': lr,
                        'weight_decay': weight_decay,
                        'warmup_epochs': warmup_epochs,
                        'epochs': epochs,
                        'batch_size': train_loader.batch_size,
                    }
                )
                # Watch model (tracks gradients and parameters)
                wandb.watch(model, log='all', log_freq=100)
                logger.info("W&B initialized successfully")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'figures'), exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.base_lr = lr
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-6
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # Mixed precision training
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using automatic mixed precision (AMP)")

    def warmup_lr(self, epoch):
        """Linear warmup for learning rate."""
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_epoch(self, epoch) -> Dict[str, float]:
        """Train for one epoch with gradient monitoring."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        n_batches = 0

        # NEW: Track gradients this epoch
        epoch_grad_norms = []

        # Warmup learning rate
        self.warmup_lr(epoch)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        for batch_idx, (patches, _) in enumerate(pbar):
            patches = patches.to(self.device, non_blocking=True)

            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    reconstruction, mask = self.model(patches, mode='reconstruction')
                    loss = reconstruction_loss(reconstruction, patches, mask)
            else:
                reconstruction, mask = self.model(patches, mode='reconstruction')
                loss = reconstruction_loss(reconstruction, patches, mask)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # NEW: Monitor gradients (every N batches)
            if batch_idx % self.log_gradients_every == 0:
                grad_stats = self.grad_monitor.compute_stats(self.model)
                if grad_stats:
                    epoch_grad_norms.append(grad_stats['grad/norm_mean'])

                    # Log to W&B
                    if self.use_wandb:
                        wandb.log({
                            'batch': epoch * len(self.train_loader) + batch_idx,
                            **grad_stats,
                            'batch_loss': loss.item()
                        })

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                from src1.training.losses import patchify_target
                target = patchify_target(patches)
                acc = compute_reconstruction_accuracy(
                    reconstruction, target, mask, threshold=1.0
                )

            # Update stats
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%',
                'avg_loss': f'{total_loss/n_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        # Store average gradient norm for this epoch
        avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0

        metrics = {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches,
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': avg_grad_norm  # NEW
        }

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        n_batches = 0

        with torch.no_grad():
            for patches, _ in tqdm(self.val_loader, desc='Validation'):
                patches = patches.to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        reconstruction, mask = self.model(patches, mode='reconstruction')
                        loss = reconstruction_loss(reconstruction, patches, mask)
                else:
                    reconstruction, mask = self.model(patches, mode='reconstruction')
                    loss = reconstruction_loss(reconstruction, patches, mask)

                from src1.training.losses import patchify_target
                target = patchify_target(patches)
                acc = compute_reconstruction_accuracy(
                    reconstruction, target, mask, threshold=1.0
                )

                total_loss += loss.item()
                total_acc += acc
                n_batches += 1

        metrics = {
            'loss': total_loss / n_batches,
            'accuracy': total_acc / n_batches
        }

        return metrics

    def visualize_reconstruction(self, epoch):
        """Visualize reconstruction on a few samples."""
        # ... (keep existing implementation)
        self.model.eval()
        patches, _ = next(iter(self.val_loader))
        patches = patches[:4].to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    reconstruction, mask = self.model(patches, mode='reconstruction')
            else:
                reconstruction, mask = self.model(patches, mode='reconstruction')

        patches_cpu = patches.cpu().numpy()
        mask_cpu = mask.cpu().numpy()

        fig, axes = plt.subplots(4, 3, figsize=(12, 12))

        for i in range(4):
            rgb_indices = [50, 100, 150]
            rgb_orig = patches_cpu[i, :, :, rgb_indices]
            if rgb_orig.shape[0] == 3:
                rgb_orig = np.transpose(rgb_orig, (1, 2, 0))
            rgb_orig = (rgb_orig - rgb_orig.min()) / (rgb_orig.max() - rgb_orig.min() + 1e-8)

            axes[i, 0].imshow(rgb_orig)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')

            mask_vis = mask_cpu[i].repeat(2, axis=0).repeat(2, axis=1)
            axes[i, 1].imshow(mask_vis, cmap='gray')
            axes[i, 1].set_title(f'Mask ({mask_cpu[i].mean():.1%})')
            axes[i, 1].axis('off')

            patch_center = patches_cpu[i, 16, 16, :]
            axes[i, 2].plot(patch_center, 'b-', alpha=0.7)
            axes[i, 2].set_title('Spectrum')
            axes[i, 2].set_xlabel('Band')
            axes[i, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'figures', f'reconstruction_epoch{epoch:03d}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def plot_training_curves(self):
        """Plot training curves including gradients."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Reconstruction Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[0, 1].plot(epochs, self.learning_rates, 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('LR Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        # NEW: Gradient norms
        if len(self.gradient_norms) > 0:
            axes[1, 0].plot(epochs[:len(self.gradient_norms)], self.gradient_norms, 'purple', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Evolution')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

        # Best val loss
        axes[1, 1].axhline(y=self.best_val_loss, color='r', linestyle='--',
                          label=f'Best: {self.best_val_loss:.4f}')
        axes[1, 1].plot(epochs, self.val_losses, 'b-', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Val Loss')
        axes[1, 1].set_title('Validation Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with gradient history."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms  # NEW
        }

        if epoch % self.save_freq == 0:
            path = os.path.join(self.save_dir, f'checkpoint_epoch{epoch:03d}.pth')
            torch.save(checkpoint, path)
            logger.info(f'Saved checkpoint: {path}')

        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f'[BEST] New best model! Val loss: {self.best_val_loss:.4f}')

    def train(self):
        """Full training loop with gradient monitoring."""
        logger.info("="*80)
        logger.info("STARTING PRE-TRAINING")
        logger.info("="*80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"W&B enabled: {self.use_wandb}")
        logger.info(f"Gradient monitoring: Every {self.log_gradients_every} batches")
        logger.info("="*80)

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.learning_rates.append(train_metrics['lr'])
            self.gradient_norms.append(train_metrics['grad_norm'])  # NEW

            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'lr': train_metrics['lr'],
                    'grad_norm': train_metrics['grad_norm']  # NEW
                })

            # Print summary
            logger.info(f'\nEpoch {epoch+1}/{self.epochs}:')
            logger.info(f'  Train Loss: {train_metrics["loss"]:.4f}  Train Acc: {train_metrics["accuracy"]:.2f}%')
            logger.info(f'  Val Loss:   {val_metrics["loss"]:.4f}  Val Acc:   {val_metrics["accuracy"]:.2f}%')
            logger.info(f'  LR: {train_metrics["lr"]:.2e}  Grad Norm: {train_metrics["grad_norm"]:.6f}')  # NEW

            # Check if best
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best=is_best)

            # Visualize
            if (epoch + 1) % 20 == 0 or is_best:
                self.visualize_reconstruction(epoch + 1)

            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()

        # Final save
        self.save_checkpoint(self.epochs, is_best=False)
        self.plot_training_curves()

        if self.use_wandb:
            wandb.finish()

        logger.info("\n" + "="*80)
        logger.info("PRE-TRAINING COMPLETE!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("="*80)