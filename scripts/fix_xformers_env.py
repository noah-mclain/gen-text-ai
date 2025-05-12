#!/usr/bin/env python3
"""
Fix environment script to replace flash-attention with xformers

This script removes flash-attention if installed and installs xformers instead.
It also generates a shell script for execution.
"""
import os
import subprocess
import sys

def check_cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch not installed. Please install PyTorch first.")
        return False

def get_cuda_version():
    """Get the CUDA version being used by PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
        return None
    except ImportError:
        return None

def generate_fix_script():
    """Generate fix script based on environment"""
    cuda_available = check_cuda_available()
    cuda_version = get_cuda_version()
    
    print(f"CUDA Available: {cuda_available}")
    if cuda_version:
        print(f"CUDA Version: {cuda_version}")
    
    with open("fix_xformers_env.sh", "w") as f:
        f.write("""#!/bin/bash
# Script to replace flash-attention with xformers
set -e  # Exit on error

# Uninstall flash-attention if present
echo "Checking for flash-attention..."
if pip list | grep -q flash-attn; then
    echo "Uninstalling flash-attention..."
    pip uninstall -y flash-attn
fi

# Install xformers
echo "Installing xformers..."
pip install xformers --no-cache-dir

# Set environment variables for stable operation
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TORCH_USE_RTLD_GLOBAL="YES"
export PYTORCH_ALLOW_NET_FALLBACK="1"

echo "Environment fixed successfully!"
echo "Please restart your training session for changes to take effect."
""")
    
    # Make script executable
    os.chmod("fix_xformers_env.sh", 0o755)

if __name__ == "__main__":
    print("Generating environment fix script...")
    generate_fix_script()
    print("\nFix script created as 'fix_xformers_env.sh'")
    print("Run 'source fix_xformers_env.sh' to apply the fixes")
    
    # Option to run the script directly
    response = input("\nWould you like to run the fix script now? (y/n): ")
    if response.lower() in ("y", "yes"):
        print("Running fix script...")
        subprocess.call(["bash", "fix_xformers_env.sh"])
    else:
        print("You can run the script later with 'source fix_xformers_env.sh'") 