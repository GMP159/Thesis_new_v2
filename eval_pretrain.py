#!/usr/bin/env python3

import zarr

zarr_path = "tests/branches_patches.zarr"

print(f"Testing: {zarr_path}")

try:
    z = zarr.open(zarr_path, mode='r')
    print(f"Opened successfully")
    print(f"Type: {type(z)}")
    print(f"Keys: {list(z.keys())}")
    
    # Try accessing domain_name
    domain = z['domain_name'][()]
    print(f"Domain: {domain}")
    
    # Try accessing patches
    patches_shape = z['patches'].shape
    print(f"Patches shape: {patches_shape}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
