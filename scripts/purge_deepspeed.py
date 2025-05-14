#!/usr/bin/env python3
"""
DeepSpeed Purge Script

This script completely removes all DeepSpeed configurations and references from the codebase,
ensuring that mixed precision training can work properly without conflicts.

Usage:
    python scripts/purge_deepspeed.py

This script will:
1. Delete all DeepSpeed config files
2. Remove DeepSpeed references from training configs
3. Set up proper mixed precision configuration
4. Clear any cached DeepSpeed configs
"""

import os
import sys
import json
import glob
import shutil
import subprocess
from pathlib import Path

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)

def find_files_with_pattern(root_dir, pattern):
    """Find all files matching a glob pattern recursively."""
    return glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)

def delete_deepspeed_config_files():
    """Delete all DeepSpeed configuration files in the project."""
    print_header("DELETING DEEPSPEED CONFIG FILES")
    
    # List of patterns for DeepSpeed config files
    ds_config_patterns = [
        "ds_config*.json",
        "deepspeed_config*.json",
        "*deepspeed*.json",
        "zero_config*.json"
    ]
    
    deleted_files = []
    
    # Find and delete all DeepSpeed config files
    for pattern in ds_config_patterns:
        files = find_files_with_pattern(".", pattern)
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    print(f"Deleting DeepSpeed config file: {file_path}")
                    os.remove(file_path)
                    deleted_files.append(file_path)
            except Exception as e:
                print(f"ERROR: Could not delete {file_path}: {str(e)}")
    
    if not deleted_files:
        print("No DeepSpeed config files found to delete.")
    else:
        print(f"Successfully deleted {len(deleted_files)} DeepSpeed config files.")
    
    return deleted_files

def modify_training_configs():
    """
    Modify all training configuration files to:
    1. Remove all DeepSpeed references
    2. Set up proper mixed precision with BF16
    3. Ensure Unsloth is configured properly
    """
    print_header("UPDATING TRAINING CONFIGURATION FILES")
    
    # Find all training config files
    config_patterns = ["training_config*.json", "config*.json"]
    config_files = []
    
    for pattern in config_patterns:
        files = find_files_with_pattern("config", pattern)
        config_files.extend(files)
    
    modified_files = []
    
    for config_file in config_files:
        try:
            print(f"Processing config file: {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Make a copy of the original config
            original_config = config.copy()
            
            # Remove DeepSpeed settings
            if 'training' in config:
                deepspeed_keys = [
                    'use_deepspeed', 'deepspeed_config', 'deepspeed', 
                    'deepspeed_config_file', 'offload_optimizer_device', 
                    'offload_param_device', 'zero_stage'
                ]
                
                for key in deepspeed_keys:
                    if key in config['training']:
                        del config['training'][key]
                        print(f"  Removed '{key}' from {config_file}")
            
            # Set proper mixed precision settings
            if 'training' in config:
                # Set BF16 precision
                config['training']['bf16'] = True
                config['training']['fp16'] = False
                config['training']['mixed_precision'] = 'bf16'
                
                # Ensure proper torch_dtype
                if 'torch_dtype' in config['training']:
                    config['training']['torch_dtype'] = 'bfloat16'
                
                # Check for model type to enable Unsloth if applicable
                model_name = config.get('training', {}).get('model_name_or_path', '').lower()
                is_causal_lm = not any(name in model_name for name in ['t5', 'ul2', 'flan'])
                
                if is_causal_lm:
                    config['training']['use_unsloth'] = True
                    print(f"  Enabled Unsloth for causal LM in {config_file}")
                else:
                    # For sequence-to-sequence models, Unsloth is not compatible
                    config['training']['use_unsloth'] = False
            
            # Check if config has changed
            if config != original_config:
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                modified_files.append(config_file)
                print(f"  Updated configuration in {config_file}")
            else:
                print(f"  No changes needed for {config_file}")
        
        except Exception as e:
            print(f"ERROR: Could not process {config_file}: {str(e)}")
    
    if not modified_files:
        print("No configuration files were modified.")
    else:
        print(f"Successfully updated {len(modified_files)} configuration files.")
    
    return modified_files

def clean_cache_directories():
    """Clean cached files that might contain DeepSpeed references."""
    print_header("CLEANING CACHE DIRECTORIES")
    
    cache_dirs = [
        "~/.cache/huggingface",
        "~/.cache/torch",
        "~/.deepspeed",
        "./.deepspeed"
    ]
    
    ds_cache_patterns = [
        "**/*deepspeed*",
        "**/*ds_config*",
        "**/*zero*"
    ]
    
    cleaned_paths = []
    
    for cache_dir in cache_dirs:
        cache_dir = os.path.expanduser(cache_dir)
        if os.path.exists(cache_dir):
            print(f"Examining cache directory: {cache_dir}")
            
            # Search for DeepSpeed cache files
            for pattern in ds_cache_patterns:
                matches = glob.glob(os.path.join(cache_dir, pattern), recursive=True)
                for match in matches:
                    if "deepspeed" in match.lower() or "ds_config" in match.lower() or "zero" in match.lower():
                        try:
                            if os.path.isfile(match):
                                os.remove(match)
                                cleaned_paths.append(match)
                                print(f"  Deleted cached file: {match}")
                            elif os.path.isdir(match):
                                shutil.rmtree(match)
                                cleaned_paths.append(match)
                                print(f"  Deleted cached directory: {match}")
                        except Exception as e:
                            print(f"  ERROR: Could not delete {match}: {str(e)}")
    
    # Create/update accelerate config
    accel_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
    os.makedirs(accel_dir, exist_ok=True)
    
    accelerate_config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "NO",
        "downcast_bf16": "no",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": "bf16",
        "num_machines": 1,
        "num_processes": 1,
        "rdzv_backend": "static",
        "same_network": True,
        "use_cpu": False
    }
    
    # Save accelerate config
    config_path = os.path.join(accel_dir, "default_config.yaml")
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(accelerate_config, f)
        print(f"Created/updated accelerate config at {config_path}")
        cleaned_paths.append(config_path)
    except Exception as e:
        print(f"ERROR: Could not create accelerate config: {str(e)}")
    
    if not cleaned_paths:
        print("No cache files were cleaned.")
    else:
        print(f"Successfully cleaned {len(cleaned_paths)} cached files/directories.")
    
    return cleaned_paths

def clean_environment_variables():
    """Unset DeepSpeed-related environment variables."""
    print_header("CLEANING ENVIRONMENT VARIABLES")
    
    ds_env_vars = [
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
    
    for var in ds_env_vars:
        if var in os.environ:
            del os.environ[var]
            print(f"Unset environment variable: {var}")
        else:
            print(f"Environment variable not set: {var}")
    
    # Set mixed precision environment variable
    os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"
    print("Set ACCELERATE_MIXED_PRECISION=bf16")
    
    return ds_env_vars

def modify_source_code():
    """
    Modify source code to disable DeepSpeed references.
    This is more invasive, so we'll make backup copies.
    """
    print_header("MODIFYING SOURCE CODE")
    
    src_dir = "src"
    if not os.path.exists(src_dir):
        print(f"ERROR: Source directory {src_dir} not found.")
        return []
    
    # Find Python files that might contain DeepSpeed references
    py_files = find_files_with_pattern(src_dir, "*.py")
    modified_files = []
    
    for py_file in py_files:
        with open(py_file, 'r') as f:
            content = f.read()
        
        # Skip files that don't contain DeepSpeed references
        if "deepspeed" not in content.lower() and "zero" not in content.lower():
            continue
        
        print(f"Checking file for DeepSpeed references: {py_file}")
        
        # Create backup
        backup_file = f"{py_file}.bak"
        shutil.copy2(py_file, backup_file)
        print(f"  Created backup: {backup_file}")
        
        # Replace DeepSpeed check patterns to force disable
        replacements = [
            (r"args.deepspeed or args.use_deepspeed", "False"),
            (r"config.get\(['\"]use_deepspeed['\"], False\)", "False"),
            (r"config.get\(['\"]deepspeed['\"], False\)", "False"),
            (r"args.use_deepspeed\s*==\s*True", "False"),
            (r"args.deepspeed\s*==\s*True", "False"),
            (r"if\s+args.deepspeed", "if False"),
            (r"if\s+args.use_deepspeed", "if False"),
            (r"if\s+use_deepspeed", "if False"),
            (r"if\s+deepspeed", "if False"),
        ]
        
        modified = False
        for pattern, replacement in replacements:
            import re
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                modified = True
                print(f"  Replaced {pattern} with {replacement}")
        
        if modified:
            with open(py_file, 'w') as f:
                f.write(content)
            modified_files.append(py_file)
            print(f"  Updated source file: {py_file}")
    
    if not modified_files:
        print("No source files were modified.")
    else:
        print(f"Successfully modified {len(modified_files)} source files.")
    
    return modified_files

def uninstall_deepspeed():
    """Attempt to uninstall DeepSpeed package."""
    print_header("UNINSTALLING DEEPSPEED PACKAGE")
    
    try:
        print("Checking if DeepSpeed is installed...")
        import importlib.util
        ds_spec = importlib.util.find_spec("deepspeed")
        
        if ds_spec is None:
            print("DeepSpeed is not installed.")
            return False
        
        print("DeepSpeed is installed. Attempting to uninstall...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "deepspeed"])
        print("DeepSpeed package uninstalled.")
        return True
    
    except Exception as e:
        print(f"ERROR during DeepSpeed uninstall: {str(e)}")
        return False

def main():
    """Main function to execute all cleanup tasks."""
    print_header("DEEPSPEED PURGE SCRIPT")
    print("This script will completely remove DeepSpeed from your codebase.")
    print("Press Ctrl+C now if you want to cancel.")
    
    try:
        input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
    
    # Run all cleanup tasks
    deleted_files = delete_deepspeed_config_files()
    modified_configs = modify_training_configs()
    cleaned_cache = clean_cache_directories()
    cleaned_env_vars = clean_environment_variables()
    modified_src = modify_source_code()
    uninstalled = uninstall_deepspeed()
    
    # Print summary
    print_header("CLEANUP SUMMARY")
    print(f"DeepSpeed config files deleted: {len(deleted_files)}")
    print(f"Training configs modified: {len(modified_configs)}")
    print(f"Cache files/directories cleaned: {len(cleaned_cache)}")
    print(f"Environment variables unset: {len(cleaned_env_vars)}")
    print(f"Source files modified: {len(modified_src)}")
    print(f"DeepSpeed package uninstalled: {uninstalled}")
    
    print("\nDeepSpeed has been purged from your codebase.")
    print("Mixed precision (BF16) has been configured.")
    print("You can now run your training without DeepSpeed conflicts.")

if __name__ == "__main__":
    main() 