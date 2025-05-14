#!/usr/bin/env python3
"""
Fix xformers Environment

This script ensures proper integration of xformers and Unsloth optimizations
for memory-efficient attention and better training performance.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_and_install_packages():
    """Check and install required packages if not already installed."""
    required_packages = {
        "xformers": "memory-efficient attention mechanisms",
        "unsloth": "optimized fine-tuning for LLMs"
    }
    
    installed = []
    missing = []
    
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            installed.append(package)
            logger.info(f"✅ {package} is already installed ({description})")
        except ImportError:
            missing.append(package)
            logger.warning(f"❌ {package} is not installed ({description})")
    
    if missing:
        logger.info(f"Installing missing packages: {', '.join(missing)}")
        for package in missing:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                logger.info(f"✅ Successfully installed {package}")
                installed.append(package)
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
    
    return installed

def set_environment_variables():
    """Set environment variables for xformers and memory-efficient attention."""
    env_vars = {
        "XFORMERS_ENABLE_FLASH_ATTN": "true",
        "XFORMERS_MEM_EFF_ATTN": "true",
        "TRANSFORMERS_USE_XFORMERS": "true"
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        logger.info(f"Set environment variable {var}={value}")
    
    # Unset any DeepSpeed variables to avoid conflicts
    deepspeed_vars = [
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE",
        "ACCELERATE_DEEPSPEED_PLUGIN_TYPE",
        "HF_DS_CONFIG",
        "DEEPSPEED_CONFIG_FILE",
        "DS_ACCELERATOR",
        "DS_OFFLOAD_PARAM",
        "DS_OFFLOAD_OPTIMIZER",
        "TRANSFORMERS_ZeRO_2_FORCE_INVALIDATE_CHECKPOINT",
        "DEEPSPEED_OVERRIDE_DISABLE"
    ]
    
    for var in deepspeed_vars:
        if var in os.environ:
            del os.environ[var]
            logger.info(f"Unset environment variable {var} to avoid conflicts")

def update_training_modules():
    """Update training modules to ensure they use xformers and Unsloth optimizations."""
    # Find training modules
    training_files = [
        project_root / "src" / "training" / "train.py",
        project_root / "src" / "training" / "trainer.py"
    ]
    
    for file_path in training_files:
        if not file_path.exists():
            logger.warning(f"Training file not found: {file_path}")
            continue
        
        logger.info(f"Updating {file_path} for xformers and Unsloth compatibility")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if we need to add xformers import
            if 'import xformers' not in content and 'from xformers' not in content:
                # Find a good place to add the import - after torch imports
                torch_import_idx = content.find('import torch')
                if torch_import_idx != -1:
                    # Find the end of the import section
                    end_of_line = content.find('\n', torch_import_idx)
                    if end_of_line != -1:
                        insert_point = end_of_line + 1
                        new_content = (
                            content[:insert_point] + 
                            '\n# Import xformers for memory-efficient attention\n'
                            'try:\n'
                            '    import xformers\n'
                            '    XFORMERS_AVAILABLE = True\n'
                            'except ImportError:\n'
                            '    XFORMERS_AVAILABLE = False\n' +
                            content[insert_point:]
                        )
                        content = new_content
                        logger.info(f"Added xformers import to {file_path}")
            
            # Fix Unsloth import order if needed
            if 'from unsloth import' in content:
                # Ensure Unsloth is imported early
                if content.find('from unsloth import') > content.find('import torch'):
                    logger.info(f"Fixing Unsloth import order in {file_path}")
                    
                    # Extract Unsloth import line
                    unsloth_import_line = ''
                    for line in content.split('\n'):
                        if 'from unsloth import' in line:
                            unsloth_import_line = line
                            break
                    
                    if unsloth_import_line:
                        # Remove the line from its current position
                        content = content.replace(unsloth_import_line + '\n', '')
                        
                        # Add it before torch import
                        torch_import_idx = content.find('import torch')
                        if torch_import_idx != -1:
                            # Find the beginning of the line
                            line_start = content.rfind('\n', 0, torch_import_idx) + 1
                            new_content = (
                                content[:line_start] + 
                                '# Import Unsloth first for proper optimizations\n' +
                                unsloth_import_line + '\n\n' +
                                content[line_start:]
                            )
                            content = new_content
                            logger.info(f"Fixed Unsloth import order in {file_path}")
            
            # Update model creation to use xformers if available
            if 'model = AutoModelForCausalLM.from_pretrained(' in content and 'attn_implementation' not in content:
                logger.info(f"Adding xformers attn_implementation to model loading in {file_path}")
                
                # Update model creation code
                content = content.replace(
                    'model = AutoModelForCausalLM.from_pretrained(',
                    'model = AutoModelForCausalLM.from_pretrained(\n        '
                    'attn_implementation="xformers" if XFORMERS_AVAILABLE else "sdpa",'
                )
                logger.info(f"Added xformers attention implementation to model loading")
            
            # Save the updated file
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"✅ Successfully updated {file_path} for xformers and Unsloth compatibility")
            
        except Exception as e:
            logger.error(f"Error updating {file_path}: {e}")

def main():
    """Main function to ensure proper xformers and Unsloth setup."""
    logger.info("Checking and fixing xformers and Unsloth integration...")
    
    # Check and install required packages
    installed_packages = check_and_install_packages()
    
    # Set environment variables
    set_environment_variables()
    
    # Update training modules
    update_training_modules()
    
    # Summary
    logger.info("✅ xformers and Unsloth environment setup complete")
    if "xformers" in installed_packages and "unsloth" in installed_packages:
        logger.info("Both xformers and Unsloth are installed and configured")
    else:
        missing = []
        if "xformers" not in installed_packages:
            missing.append("xformers")
        if "unsloth" not in installed_packages:
            missing.append("unsloth")
        logger.warning(f"⚠️ Some packages could not be installed: {', '.join(missing)}")
        logger.warning("Performance might be reduced. Check the logs for details.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 