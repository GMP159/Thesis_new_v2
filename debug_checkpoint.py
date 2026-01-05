"""
Quick checkpoint debugger - prints all keys
"""

import torch

checkpoint_path = r'D:\Thesis_new_v2\outputs\checkpoints\pretrain\checkpoint_best.pth'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("\n" + "="*80)
print("CHECKPOINT STRUCTURE")
print("="*80)

# Top-level keys
print("\nTop-level keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Model state dict keys (first 20)
print("\n" + "="*80)
print("MODEL STATE DICT KEYS (first 30):")
print("="*80)
model_keys = list(checkpoint['model_state_dict'].keys())
for i, key in enumerate(model_keys[:30]):
    shape = checkpoint['model_state_dict'][key].shape if hasattr(checkpoint['model_state_dict'][key], 'shape') else 'N/A'
    print(f"{i+1:3d}. {key:60s} {shape}")

if len(model_keys) > 30:
    print(f"\n... and {len(model_keys) - 30} more keys")

# Search for classification-related keys
print("\n" + "="*80)
print("CLASSIFICATION/HEAD KEYS:")
print("="*80)
classif_keys = [k for k in model_keys if 'classif' in k.lower() or 'head' in k.lower()]
if classif_keys:
    for key in classif_keys:
        shape = checkpoint['model_state_dict'][key].shape
        print(f"  {key}: {shape}")
else:
    print("  No classification keys found!")

# Try to find num_classes
print("\n" + "="*80)
print("FINDING NUM_CLASSES:")
print("="*80)

possible_keys = [
    'classification_head.classifier.weight',
    'classification_head.weight',
    'module.classification_head.classifier.weight',  # DDP wrapping
    'module.classification_head.weight',
    'domain_head.weight',
    'head.weight',
    'fc.weight'
]

found = False
for key in possible_keys:
    if key in checkpoint['model_state_dict']:
        shape = checkpoint['model_state_dict'][key].shape
        num_classes = shape[0]
        print(f" FOUND: {key}")
        print(f"   Shape: {shape}")
        print(f"   num_classes = {num_classes}")
        found = True
        break

if not found:
    print(" None of the standard keys found")
    print("\nSearching for ANY weight with shape [N, ...]:")
    for key in model_keys:
        if 'weight' in key:
            val = checkpoint['model_state_dict'][key]
            if hasattr(val, 'shape') and len(val.shape) >= 2:
                print(f"  {key}: {val.shape}")