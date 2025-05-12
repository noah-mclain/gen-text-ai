#!/usr/bin/env python3
"""
Quick NumPy and CUDA fix for Paperspace

This script creates a quick fix for NumPy compatibility issues and sets CUDA env vars.
"""

with open("quick_fix.sh", "w") as f:
    f.write("""#!/bin/bash
# Quick fix script for NumPy and CUDA
echo "Fixing NumPy compatibility issue..."
pip install numpy==1.24.3 -q

# Set CUDA environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TORCH_USE_RTLD_GLOBAL="YES"
export PYTORCH_ALLOW_NET_FALLBACK="1"
export TORCH_CUDA_ARCH_LIST="8.6"

echo "Environment fixes applied successfully"
""")

import os
os.chmod("quick_fix.sh", 0o755)

print("Created quick_fix.sh")
print("Run 'source quick_fix.sh' to apply the fixes") 