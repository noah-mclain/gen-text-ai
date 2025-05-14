#!/usr/bin/env python3
"""
Verify the Paperspace environment and GPU configuration
"""

import os
import sys
import subprocess
import torch
import platform

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.check_output(cmd, shell=True, text=True)
        return result.strip()
    except Exception as e:
        return f"Error running {cmd}: {str(e)}"

def check_environment():
    """Check the environment"""
    print("===== SYSTEM INFO =====")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    print("\n===== NVIDIA GPU INFO =====")
    print(run_cmd("nvidia-smi"))
    
    print("\n===== PYTORCH INFO =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
    
    print("\n===== DEEPSPEED ENV =====")
    print(f"ACCELERATE_USE_DEEPSPEED: {os.environ.get('ACCELERATE_USE_DEEPSPEED', 'Not set')}")
    print(f"ACCELERATE_DEEPSPEED_CONFIG_FILE: {os.environ.get('ACCELERATE_DEEPSPEED_CONFIG_FILE', 'Not set')}")
    print(f"HF_DS_CONFIG: {os.environ.get('HF_DS_CONFIG', 'Not set')}")
    print(f"ACCELERATE_DEEPSPEED_PLUGIN_TYPE: {os.environ.get('ACCELERATE_DEEPSPEED_PLUGIN_TYPE', 'Not set')}")
    
    print("\n===== UNSLOTH INFO =====")
    try:
        import unsloth
        print(f"Unsloth version: {unsloth.__version__}")
        device_type = getattr(unsloth, "DEVICE_TYPE", "Unknown")
        print(f"Unsloth device type: {device_type}")
    except ImportError:
        print("Unsloth not installed")
    except Exception as e:
        print(f"Error checking Unsloth: {str(e)}")

if __name__ == "__main__":
    check_environment()
    sys.exit(0) 