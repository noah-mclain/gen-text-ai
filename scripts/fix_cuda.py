#!/usr/bin/env python3
"""
CUDA/PyTorch Environment Fixer

This script diagnoses and attempts to fix common CUDA and PyTorch compatibility issues
often encountered when running on Paperspace, Lambda Labs, or other cloud GPU providers.

Usage:
    python scripts/fix_cuda.py

The script will output recommended environment variables to set, and if run with --apply,
will automatically set them for the current session.
"""

import os
import sys
import subprocess
import argparse
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_nvidia_info():
    """Get detailed NVIDIA GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"nvidia-smi returned error code {result.returncode}\n{result.stderr}"
    except Exception as e:
        return f"Error running nvidia-smi: {str(e)}"

def get_cuda_version():
    """Get CUDA version info from nvcc if available."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            return f"nvcc returned error code {result.returncode}\n{result.stderr}"
    except Exception as e:
        return f"Error running nvcc: {str(e)}"

def get_system_info():
    """Get basic system information."""
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
    }
    
    # Get environment variables related to CUDA
    cuda_vars = {}
    for key, value in os.environ.items():
        if "CUDA" in key or "TORCH" in key:
            cuda_vars[key] = value
    
    info["cuda_env_vars"] = cuda_vars
    
    return info

def check_pytorch():
    """Check PyTorch installation and CUDA compatibility."""
    try:
        import torch
        
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_name"] = torch.cuda.get_device_name(0)
            
            try:
                info["cudnn_version"] = torch.backends.cudnn.version()
                info["cudnn_enabled"] = torch.backends.cudnn.enabled
            except:
                info["cudnn_error"] = "Could not determine cuDNN info"
        
        return info
    except ImportError:
        return {"error": "PyTorch not installed"}
    except Exception as e:
        return {"error": f"Error checking PyTorch: {str(e)}"}

def get_recommended_env_vars(pytorch_info):
    """Get recommended environment variables based on the system and PyTorch info."""
    
    # Start with safe defaults
    recommended = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        "TORCH_USE_RTLD_GLOBAL": "YES",
        "PYTORCH_ALLOW_NET_FALLBACK": "1",
    }
    
    # Check if PyTorch can see CUDA
    if isinstance(pytorch_info, dict) and not pytorch_info.get("cuda_available", False):
        # If CUDA isn't available to PyTorch, check if NVIDIA driver is present
        nvidia_info = get_nvidia_info()
        if "NVIDIA-SMI" in nvidia_info and "Driver Version" in nvidia_info:
            # NVIDIA driver is present, but PyTorch can't see it
            # This suggests environment/library issues
            recommended.update({
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}",
                "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
                "TORCH_NVCC_FLAGS": "-Xfatbin -compress-all"
            })
    
    # If using A6000 or other Ampere GPUs
    if isinstance(pytorch_info, dict) and pytorch_info.get("device_name", "").lower().find("a6000") >= 0:
        recommended["TORCH_CUDA_ARCH_LIST"] = "8.6"
    
    return recommended

def main():
    parser = argparse.ArgumentParser(description="Fix CUDA and PyTorch compatibility issues")
    parser.add_argument("--apply", action="store_true", help="Apply recommended fixes to current environment")
    parser.add_argument("--verbose", action="store_true", help="Show detailed diagnostic information")
    args = parser.parse_args()
    
    print("\n===== CUDA/PyTorch Environment Checker =====\n")
    
    # Get system information
    system_info = get_system_info()
    pytorch_info = check_pytorch()
    
    # Print basic info
    print(f"Python: {system_info['python'].split()[0]}")
    print(f"Platform: {system_info['platform']}")
    
    # Print PyTorch info
    if "error" in pytorch_info:
        print(f"PyTorch: {pytorch_info['error']}")
    else:
        print(f"PyTorch: {pytorch_info.get('torch_version', 'unknown')}")
        print(f"CUDA available: {pytorch_info.get('cuda_available', False)}")
        if pytorch_info.get('cuda_available', False):
            print(f"CUDA version: {pytorch_info.get('cuda_version', 'unknown')}")
            print(f"GPU: {pytorch_info.get('device_name', 'unknown')}")
            print(f"cuDNN version: {pytorch_info.get('cudnn_version', 'unknown')}")
    
    # Print detailed info if verbose
    if args.verbose:
        print("\n----- Detailed NVIDIA Information -----")
        print(get_nvidia_info())
        
        print("\n----- CUDA Compiler Information -----")
        print(get_cuda_version())
        
        print("\n----- Current CUDA/PyTorch Environment Variables -----")
        for key, value in system_info["cuda_env_vars"].items():
            print(f"{key}={value}")
    
    # Get recommended environment variables
    recommended_vars = get_recommended_env_vars(pytorch_info)
    
    print("\n----- Recommended Environment Variables -----")
    for key, value in recommended_vars.items():
        print(f"export {key}={value}")
    
    # Create a script with the environment variables
    with open("cuda_env_setup.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Environment variables for CUDA/PyTorch compatibility\n")
        f.write("# Generated by fix_cuda.py\n\n")
        
        for key, value in recommended_vars.items():
            f.write(f"export {key}={value}\n")
    
    os.chmod("cuda_env_setup.sh", 0o755)
    print("\nCreated cuda_env_setup.sh with recommended environment variables")
    print("Run `source cuda_env_setup.sh` to apply them to your current session")
    
    # Apply the recommendations if requested
    if args.apply:
        print("\nApplying recommended environment variables...")
        for key, value in recommended_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")
        
        print("\nAttempting to import PyTorch again...")
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"Error importing PyTorch: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 