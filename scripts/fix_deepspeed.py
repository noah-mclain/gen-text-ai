#!/usr/bin/env python3
"""
Fix DeepSpeed Configuration

This script ensures that a valid DeepSpeed configuration file is placed in all
potential locations where the training scripts might look for it.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default DeepSpeed config with ZeRO optimization
DEFAULT_DS_CONFIG = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "steps_per_print": 50,
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "wall_clock_breakdown": False,
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-4,
            "warmup_num_steps": 100,
            "total_num_steps": 10000
        }
    }
}

def ensure_valid_deepspeed_config(config_path):
    """Ensure that a valid DeepSpeed config exists at the specified path."""
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if os.path.exists(config_path):
            try:
                # Read the existing config
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                
                # Check if it has the zero_optimization section
                if "zero_optimization" not in existing_config:
                    logger.warning(f"Config at {config_path} is missing zero_optimization. Adding it.")
                    existing_config["zero_optimization"] = DEFAULT_DS_CONFIG["zero_optimization"]
                    
                    # Write the updated config back
                    with open(config_path, 'w') as f:
                        json.dump(existing_config, f, indent=2)
                else:
                    logger.info(f"Existing config at {config_path} has zero_optimization. No update needed.")
                    
                return True
            except Exception as e:
                logger.error(f"Error reading or updating config at {config_path}: {e}")
                logger.info(f"Creating a new config at {config_path}")
                with open(config_path, 'w') as f:
                    json.dump(DEFAULT_DS_CONFIG, f, indent=2)
                return True
        else:
            # Create a new config file
            logger.info(f"Creating DeepSpeed config at {config_path}")
            with open(config_path, 'w') as f:
                json.dump(DEFAULT_DS_CONFIG, f, indent=2)
            return True
            
    except Exception as e:
        logger.error(f"Failed to create/update DeepSpeed config at {config_path}: {e}")
        return False

def fix_deepspeed_config():
    """Place DeepSpeed config files at all potential locations."""
    # Get the current working directory and project root
    cwd = os.getcwd()
    project_root = Path(__file__).parent.parent
    
    # All potential paths where the config might be needed
    potential_paths = [
        os.path.join(cwd, "ds_config_a6000.json"),
        os.path.join(project_root, "ds_config_a6000.json"),
        "/notebooks/ds_config_a6000.json" if os.path.exists("/notebooks") else None
    ]
    
    # Filter out None values
    potential_paths = [p for p in potential_paths if p]
    
    # Get the environment variable if it's set
    env_path = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE")
    if env_path:
        potential_paths.append(env_path)
    
    # Ensure valid config at each path
    success_path = None
    for path in potential_paths:
        if path:
            try:
                if ensure_valid_deepspeed_config(path):
                    logger.info(f"✅ DeepSpeed config set up at: {path}")
                    success_path = success_path or path
            except Exception as e:
                logger.error(f"Failed to set up DeepSpeed config at {path}: {e}")
    
    # Use the first successful path as our primary config
    primary_config_path = success_path or potential_paths[0]
    
    # Explicitly set all required environment variables for DeepSpeed
    # This is crucial to fix the 'NoneType' object has no attribute 'hf_ds_config' error
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = primary_config_path
    os.environ["HF_DS_CONFIG"] = primary_config_path
    os.environ["ACCELERATE_DEEPSPEED_PLUGIN_TYPE"] = "deepspeed"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    
    logger.info(f"Set ACCELERATE_DEEPSPEED_CONFIG_FILE to {primary_config_path}")
    logger.info(f"Set HF_DS_CONFIG to {primary_config_path}")
    logger.info("Set ACCELERATE_DEEPSPEED_PLUGIN_TYPE to 'deepspeed'")
    logger.info("Set ACCELERATE_USE_DEEPSPEED to 'true'")
    
    # Also add to models directory for completeness
    models_dir = os.path.join(project_root, "models")
    if os.path.exists(models_dir):
        models_config_path = os.path.join(models_dir, "ds_config.json")
        if ensure_valid_deepspeed_config(models_config_path):
            logger.info(f"✅ DeepSpeed config set up at: {models_config_path}")
    
    if os.path.exists("/notebooks"):
        os.makedirs("/notebooks/models", exist_ok=True)
        models_config_path = "/notebooks/models/ds_config.json"
        if ensure_valid_deepspeed_config(models_config_path):
            logger.info(f"✅ DeepSpeed config set up at: {models_config_path}")

if __name__ == "__main__":
    logger.info("Starting DeepSpeed config fix")
    fix_deepspeed_config()
    logger.info("DeepSpeed config fix completed")
    
    # Note for the user
    print("\n" + "=" * 60)
    print("✅ DeepSpeed configuration has been fixed.")
    print("DeepSpeed configs with proper ZeRO settings have been placed in all potential locations.")
    print("\nRun the following on a fresh shell to ensure the environment is set correctly:")
    primary_path = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE", os.path.join(os.getcwd(), "ds_config_a6000.json"))
    print(f"export ACCELERATE_DEEPSPEED_CONFIG_FILE={primary_path}")
    print(f"export HF_DS_CONFIG={primary_path}")
    print("export ACCELERATE_DEEPSPEED_PLUGIN_TYPE=deepspeed")
    print("export ACCELERATE_USE_DEEPSPEED=true")
    print("=" * 60 + "\n") 