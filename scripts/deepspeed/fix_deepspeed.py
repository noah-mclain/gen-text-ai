#!/usr/bin/env python3
"""
Fix DeepSpeed configuration for A6000 GPUs.
This script ensures that DeepSpeed is correctly configured for use with transformers and accelerate.
"""

import os
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_deepspeed_config():
    """
    Fix the DeepSpeed configuration to work properly with transformers and accelerate.
    - Create or validate ds_config_a6000.json
    - Set environment variables 
    - Create symlinks for better detection
    """
    # Find the project root directory
    project_root = Path(__file__).resolve().parent.parent
    logger.info(f"Project root: {project_root}")
    
    # Check if we're in Paperspace
    in_paperspace = os.path.exists("/notebooks")
    if in_paperspace:
        logger.info("Paperspace environment detected")
    
    # Possible locations for the DeepSpeed config
    config_paths = [
        project_root / "ds_config_a6000.json",  # Project root
        project_root / "deepspeed_config.json",  # Alternative name
        Path("/notebooks/ds_config_a6000.json") if in_paperspace else None,  # Paperspace notebook dir
        project_root / "config" / "ds_config_zero3.json",  # Config dir
        project_root / "models" / "ds_config.json",  # Models dir
    ]
    
    # Filter out None values
    config_paths = [p for p in config_paths if p is not None]
    
    # Check existing configs
    existing_config = None
    for path in config_paths:
        if path.exists():
            logger.info(f"Existing config found at: {path}")
            try:
                with open(path, 'r') as f:
                    config_data = json.load(f)
                if "zero_optimization" in config_data:
                    existing_config = path
                    logger.info(f"Existing config at {path} has zero_optimization. No update needed.")
                    break
            except json.JSONDecodeError:
                logger.warning(f"File at {path} is not valid JSON.")
            except Exception as e:
                logger.warning(f"Error reading {path}: {e}")
    
    # If no valid config is found, create one
    if existing_config is None:
        default_config = {
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
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
                "reduce_scatter": True
            },
            "gradient_accumulation_steps": 16,
            "gradient_clipping": 1.0,
            "train_batch_size": 32,
            "train_micro_batch_size_per_gpu": 2
        }
        
        # Use the first path as default
        config_path = config_paths[0]
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"✅ Created new DeepSpeed config at: {config_path}")
            existing_config = config_path
        except Exception as e:
            logger.error(f"Failed to create config at {config_path}: {e}")
            return False
    
    # Create symbolic links for better detection
    symlink_paths = [
        project_root / "deepspeed_config.json",  # Root dir with standard name
        Path("/notebooks/deepspeed_config.json") if in_paperspace else None,  # Paperspace notebook dir
        project_root / "models" / "deepspeed_config.json" if (project_root / "models").exists() else None,  # Models dir
    ]
    
    # Filter out None values
    symlink_paths = [p for p in symlink_paths if p is not None]
    
    # Create symlinks for better detection
    for path in symlink_paths:
        if not path.exists():
            try:
                # Remove existing symlink if broken
                if path.is_symlink():
                    path.unlink()
                
                # Create relative symlink
                path.symlink_to(existing_config)
                logger.info(f"✅ Created symlink at {path} pointing to {existing_config}")
            except Exception as e:
                logger.warning(f"Failed to create symlink at {path}: {e}")
    
    # Set environment variables
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = str(existing_config)
    os.environ["HF_DS_CONFIG"] = str(existing_config)
    os.environ["ACCELERATE_DEEPSPEED_PLUGIN_TYPE"] = "deepspeed"
    os.environ["DEEPSPEED_CONFIG_FILE"] = str(existing_config)  # For newer versions
    
    logger.info(f"Set ACCELERATE_USE_DEEPSPEED to 'true'")
    logger.info(f"Set ACCELERATE_DEEPSPEED_CONFIG_FILE to {existing_config}")
    logger.info(f"Set HF_DS_CONFIG to {existing_config}")
    logger.info(f"Set ACCELERATE_DEEPSPEED_PLUGIN_TYPE to 'deepspeed'")
    logger.info(f"Set DEEPSPEED_CONFIG_FILE to {existing_config}")
    
    # Copy to Paperspace notebooks directory if we're in that environment
    if in_paperspace and str(existing_config) != "/notebooks/ds_config_a6000.json":
        try:
            import shutil
            shutil.copy(existing_config, "/notebooks/ds_config_a6000.json")
            logger.info("✅ Copied DeepSpeed config to /notebooks/ds_config_a6000.json")
            
            # Also copy to models dir in notebooks
            os.makedirs("/notebooks/models", exist_ok=True)
            shutil.copy(existing_config, "/notebooks/models/ds_config.json")
            logger.info("✅ Copied DeepSpeed config to /notebooks/models/ds_config.json")
        except Exception as e:
            logger.warning(f"Failed to copy config to Paperspace notebooks directory: {e}")
    
    logger.info("DeepSpeed config fix completed")
    return True

if __name__ == "__main__":
    print("\n============================================================")
    if fix_deepspeed_config():
        print("✅ DeepSpeed configuration has been fixed.")
        print("DeepSpeed configs with proper ZeRO settings have been placed in all potential locations.")
        
        print("\nRun the following on a fresh shell to ensure the environment is set correctly:")
        print(f"export ACCELERATE_DEEPSPEED_CONFIG_FILE={os.environ.get('ACCELERATE_DEEPSPEED_CONFIG_FILE')}")
        print(f"export HF_DS_CONFIG={os.environ.get('HF_DS_CONFIG')}")
        print(f"export ACCELERATE_DEEPSPEED_PLUGIN_TYPE=deepspeed")
        print(f"export ACCELERATE_USE_DEEPSPEED=true")
    else:
        print("❌ Failed to fix DeepSpeed configuration.")
    print("============================================================\n") 