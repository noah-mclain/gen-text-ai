#!/usr/bin/env python3
"""
Clean up any DeepSpeed environment variables that might cause conflicts
with standard training (no DeepSpeed).
"""

import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_deepspeed_env():
    """
    Detect and unset DeepSpeed-related environment variables
    """
    # List of DeepSpeed environment variables to check and unset
    deepspeed_vars = [
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE",
        "ACCELERATE_DEEPSPEED_PLUGIN_TYPE",
        "HF_DS_CONFIG",
        "DEEPSPEED_CONFIG_FILE",
        "DS_ACCELERATOR",
        "DS_OFFLOAD_PARAM",
        "DS_OFFLOAD_OPTIMIZER",
        "TRANSFORMERS_ZeRO_2_FORCE_INVALIDATE_CHECKPOINT"
    ]
    
    # Check and unset each variable
    cleaned = False
    for var in deepspeed_vars:
        if var in os.environ:
            logger.info(f"Removing environment variable: {var}={os.environ[var]}")
            del os.environ[var]
            cleaned = True
    
    # Look for DeepSpeed files and warn
    potential_config_paths = [
        "ds_config_a6000.json",
        "deepspeed_config.json", 
        "config/ds_config_zero3.json", 
        "models/ds_config.json",
        "/notebooks/ds_config_a6000.json"
    ]
    
    # Report any found config files (but don't delete them)
    for path in potential_config_paths:
        if os.path.exists(path):
            logger.warning(f"DeepSpeed config file found at {path} - will be ignored")
    
    if cleaned:
        logger.info("Successfully cleaned DeepSpeed environment variables")
    else:
        logger.info("No DeepSpeed environment variables found to clean")
    
    # Also unset in current process and parent process
    print("Copy and paste these commands to clean your shell environment:")
    for var in deepspeed_vars:
        print(f"unset {var}")
    
    return cleaned

if __name__ == "__main__":
    print("\n============================================================")
    print(" DEEPSPEED ENVIRONMENT CLEANUP ")
    print("============================================================\n")
    
    cleaned = clean_deepspeed_env()
    
    print("\n============================================================")
    print(f"{'✅ Cleanup performed' if cleaned else '✓ No cleanup needed'}")
    print("============================================================\n") 