#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import logging
from pathlib import Path
import sys

project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src1.models.masked_sst import create_model
from src1.data.dataset import MultiDomainHSIDataset, find_zarr_files
from src1.data.transforms import get_val_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def extract_features_and_train_head(encoder, dataset, num_classes, num_epochs=10, 
                                     train_ratio=0.8, device='cuda'):
    
    # Split dataset
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Extract features with frozen encoder
    logger.info("Extracting training features...")
    encoder.eval()
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for patches, labels in tqdm(train_loader):
            patches = patches.to(device)
            features, _ = encoder.forward_encoder(patches, apply_masking=False)
            features_pooled = features.mean(dim=(1, 2))
            train_features.append(features_pooled)
            train_labels.append(labels)
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0).to(device)
    
    logger.info("Extracting validation features...")
    val_features = []
    val_labels = []
    
    with torch.no_grad():
        for patches, labels in tqdm(val_loader):
            patches = patches.to(device)
            features, _ = encoder.forward_encoder(patches, apply_masking=False)
            features_pooled = features.mean(dim=(1, 2))
            val_features.append(features_pooled)
            val_labels.append(labels)
    
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0).to(device)
    
    # Train classification head
    feature_dim = train_features.shape[1]
    head = LinearClassificationHead(feature_dim, num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    
    logger.info(f"Training classification head for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        head.train()
        
        # Train
        perm = torch.randperm(len(train_features))
        train_loss = 0.0
        train_correct = 0
        
        for i in range(0, len(train_features), 64):
            idx = perm[i:i+64]
            batch_features = train_features[idx]
            batch_labels = train_labels[idx]
            
            optimizer.zero_grad()
            outputs = head(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_labels).sum().item()
        
        train_acc = train_correct / len(train_features) * 100
        
        # Validate
        head.eval()
        with torch.no_grad():
            val_outputs = head(val_features)
            val_loss = criterion(val_outputs, val_labels).item()
            val_correct = (val_outputs.argmax(1) == val_labels).sum().item()
            val_acc = val_correct / len(val_features) * 100
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
    
    return best_val_acc


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='tests')
    parser.add_argument('--test_domain', type=str, default='branches')
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test domain dataset
    logger.info(f"Loading {args.test_domain} dataset...")
    zarr_paths = find_zarr_files(args.data_root, [args.test_domain])
    dataset = MultiDomainHSIDataset(
        zarr_paths=zarr_paths,
        split='train',
        train_ratio=1.0,
        return_domain_labels=True,
        normalize_per_patch=True,
        transform=get_val_transforms()
    )
    
    num_classes = len(dataset.domain_to_id)
    logger.info(f"Dataset: {len(dataset)} samples, {num_classes} classes")
    
    # Test 1: Pretrained encoder
    logger.info("\n" + "="*80)
    logger.info("TEST 1: PRETRAINED ENCODER + LINEAR HEAD")
    logger.info("="*80)
    
    pretrained_model = create_model(num_classes=8).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    pretrained_model.load_state_dict(state_dict)
    pretrained_model.eval()
    
    pretrained_acc = extract_features_and_train_head(
        pretrained_model, dataset, num_classes, args.num_epochs, device=device
    )
    
    # Test 2: Random encoder
    logger.info("\n" + "="*80)
    logger.info("TEST 2: RANDOM ENCODER + LINEAR HEAD")
    logger.info("="*80)
    
    random_model = create_model(num_classes=8).to(device)
    random_model.eval()
    
    random_acc = extract_features_and_train_head(
        random_model, dataset, num_classes, args.num_epochs, device=device
    )
    
    # Results
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Pretrained encoder: {pretrained_acc:.2f}%")
    logger.info(f"Random encoder: {random_acc:.2f}%")
    logger.info(f"Improvement: {pretrained_acc - random_acc:.2f}%")
    logger.info("="*80)
    
    if pretrained_acc > random_acc + 5:
        logger.info("Pretraining HELPS - learned useful features!")
    elif pretrained_acc < random_acc - 5:
        logger.info("Pretraining HURTS - features are domain-specific!")
    else:
        logger.info("Pretraining has MINIMAL effect")


if __name__ == '__main__':
    main()
