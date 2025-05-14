#!/usr/bin/env python3
"""
Test DeepSpeed Configuration

This script checks if the DeepSpeed configuration is properly set up in the environment
and tests whether it's compatible with the transformers library.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to pythonpath
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_deepspeed_environment():
    """Check if DeepSpeed environment variables are properly set."""
    required_vars = [
        "ACCELERATE_USE_DEEPSPEED",
        "ACCELERATE_DEEPSPEED_CONFIG_FILE",
        "ACCELERATE_DEEPSPEED_PLUGIN_TYPE",
        "HF_DS_CONFIG"
    ]
    
    all_present = True
    logger.info("=== DeepSpeed Environment Variables ===")
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"✅ {var} = {value}")
        else:
            logger.warning(f"❌ {var} is not set")
            all_present = False
    
    return all_present

def check_config_files():
    """Check if DeepSpeed config files exist and are valid JSON."""
    config_path = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE")
    hf_config_path = os.environ.get("HF_DS_CONFIG")
    
    logger.info("=== DeepSpeed Config Files ===")
    
    all_valid = True
    for name, path in [("ACCELERATE_DEEPSPEED_CONFIG_FILE", config_path), 
                      ("HF_DS_CONFIG", hf_config_path)]:
        if not path:
            logger.warning(f"❌ {name} path is not set")
            all_valid = False
            continue
            
        if not os.path.exists(path):
            logger.warning(f"❌ {name} points to non-existent file: {path}")
            all_valid = False
            continue
            
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                
            if "zero_optimization" not in config:
                logger.warning(f"❌ {name} is missing zero_optimization section")
                all_valid = False
            else:
                logger.info(f"✅ {name} is a valid JSON with zero_optimization (stage: {config['zero_optimization'].get('stage', 'not set')})")
                
        except json.JSONDecodeError:
            logger.warning(f"❌ {name} contains invalid JSON")
            all_valid = False
        except Exception as e:
            logger.warning(f"❌ Error reading {name}: {e}")
            all_valid = False
            
    return all_valid

def check_transformers_deepspeed_integration():
    """Check if transformers can correctly load the DeepSpeed config."""
    try:
        import transformers
        
        logger.info("=== Transformers DeepSpeed Integration ===")
        logger.info(f"Transformers version: {transformers.__version__}")
        
        if not hasattr(transformers.integrations, "is_deepspeed_available"):
            logger.warning("❌ Your transformers version doesn't have DeepSpeed integration checks")
            return False
            
        if not transformers.integrations.is_deepspeed_available():
            logger.warning("❌ DeepSpeed is not available according to transformers")
            return False
        
        logger.info("✅ DeepSpeed is available according to transformers")
        
        # Try to access the HF DeepSpeed Config utility function if it exists
        if hasattr(transformers.deepspeed, "HfDeepSpeedConfig"):
            try:
                hf_ds_config_path = os.environ.get("HF_DS_CONFIG")
                if hf_ds_config_path and os.path.exists(hf_ds_config_path):
                    config = transformers.deepspeed.HfDeepSpeedConfig(hf_ds_config_path)
                    logger.info("✅ transformers.deepspeed.HfDeepSpeedConfig loaded successfully")
                    return True
                else:
                    logger.warning(f"❌ HF_DS_CONFIG not set or file does not exist")
                    return False
            except Exception as e:
                logger.warning(f"❌ Error loading HfDeepSpeedConfig: {e}")
                return False
        else:
            logger.info("ℹ️ transformers.deepspeed.HfDeepSpeedConfig not available in this version")
            
        # For older versions, try to see if DeepSpeed plugin works
        try:
            from accelerate import Accelerator
            accelerator = Accelerator()
            logger.info(f"✅ Accelerator created successfully with state: {accelerator.state}")
            return True
        except Exception as e:
            logger.warning(f"❌ Error creating Accelerator: {e}")
            return False
            
    except ImportError:
        logger.warning("❌ transformers package is not installed")
        return False

def fix_environment():
    """Attempt to fix the DeepSpeed environment if issues are detected."""
    logger.info("=== Attempting to fix DeepSpeed Environment ===")
    
    # Find potential DeepSpeed config files
    potential_paths = [
        os.path.join(os.getcwd(), "ds_config_a6000.json"),
        os.path.join(project_root, "ds_config_a6000.json"),
        "/notebooks/ds_config_a6000.json" if os.path.exists("/notebooks") else None,
        os.path.join(project_root, "config", "ds_config_zero3.json"),
        os.path.join(project_root, "models", "ds_config.json")
    ]
    
    # Filter out None values
    potential_paths = [p for p in potential_paths if p and os.path.exists(p)]
    
    if potential_paths:
        config_path = potential_paths[0]
        logger.info(f"Found potential DeepSpeed config at: {config_path}")
        
        # Set environment variables
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
        os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = config_path
        os.environ["HF_DS_CONFIG"] = config_path
        os.environ["ACCELERATE_DEEPSPEED_PLUGIN_TYPE"] = "deepspeed"
        
        logger.info(f"Set ACCELERATE_USE_DEEPSPEED to 'true'")
        logger.info(f"Set ACCELERATE_DEEPSPEED_CONFIG_FILE to {config_path}")
        logger.info(f"Set HF_DS_CONFIG to {config_path}")
        logger.info(f"Set ACCELERATE_DEEPSPEED_PLUGIN_TYPE to 'deepspeed'")
        
        return True
    else:
        # Try to use the fix_deepspeed.py script if available
        fix_script = os.path.join(project_root, "scripts", "fix_deepspeed.py")
        if os.path.exists(fix_script):
            logger.info(f"Running DeepSpeed fix script: {fix_script}")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("fix_deepspeed", fix_script)
                fix_deepspeed = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(fix_deepspeed)
                fix_deepspeed.fix_deepspeed_config()
                return True
            except Exception as e:
                logger.error(f"Error running fix script: {e}")
                return False
        else:
            logger.warning("No DeepSpeed config files found and no fix script available")
            return False

def main():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print(" DEEPSPEED CONFIGURATION TEST ")
    print("=" * 60 + "\n")
    
    env_ok = check_deepspeed_environment()
    config_ok = check_config_files()
    integration_ok = check_transformers_deepspeed_integration()
    
    print("\n" + "=" * 60)
    print(" SUMMARY ")
    print("=" * 60)
    print(f"Environment variables: {'✅ OK' if env_ok else '❌ ISSUES FOUND'}")
    print(f"Config files: {'✅ OK' if config_ok else '❌ ISSUES FOUND'}")
    print(f"Transformers integration: {'✅ OK' if integration_ok else '❌ ISSUES FOUND'}")
    
    if not env_ok or not config_ok:
        print("\nAttempting to fix DeepSpeed configuration...")
        fixed = fix_environment()
        
        if fixed:
            print("\nFixed environment. Running tests again...\n")
            env_ok = check_deepspeed_environment()
            config_ok = check_config_files()
            integration_ok = check_transformers_deepspeed_integration()
            
            print("\n" + "=" * 60)
            print(" UPDATED SUMMARY ")
            print("=" * 60)
            print(f"Environment variables: {'✅ OK' if env_ok else '❌ ISSUES FOUND'}")
            print(f"Config files: {'✅ OK' if config_ok else '❌ ISSUES FOUND'}")
            print(f"Transformers integration: {'✅ OK' if integration_ok else '❌ ISSUES FOUND'}")
    
    if env_ok and config_ok and integration_ok:
        print("\n✅ DeepSpeed is properly configured")
        return 0
    else:
        print("\n❌ DeepSpeed configuration has issues")
        print("\nTo fix, run:")
        print("python scripts/fix_deepspeed.py")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 