#!/usr/bin/env python3
"""
Verify Paths Utility

This script verifies that all critical paths and configurations are 
properly set up in the current environment.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_project_root():
    """Add project root to Python path."""
    # Get current file's location
    current_file = os.path.abspath(__file__)
    
    # Navigate up to project root (assuming script is in scripts/utilities)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    # Add to sys.path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to Python path: {project_root}")
    
    return project_root

def check_config_files():
    """Check configuration files."""
    project_root = add_project_root()
    
    config_files = [
        "config/dataset_config.json",
        "config/training_config.json"
    ]
    
    results = {}
    
    for config_file in config_files:
        # Try multiple possible locations
        config_paths_to_try = [
            config_file,  # Relative to current directory
            os.path.join(project_root, config_file),  # Relative to project root
            os.path.abspath(config_file),  # Absolute path
            os.path.join('/notebooks', config_file)  # For Paperspace
        ]
        
        found = False
        for path in config_paths_to_try:
            if os.path.exists(path):
                logger.info(f"✓ Found configuration file: {path}")
                
                # Try to load the file to verify it's valid JSON
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"  ✓ Successfully loaded {len(config)} items from {path}")
                    found = True
                    results[config_file] = {
                        "path": path,
                        "valid": True,
                        "items": len(config)
                    }
                    break
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {path}: {e}")
                    results[config_file] = {
                        "path": path,
                        "valid": False,
                        "error": str(e)
                    }
        
        if not found:
            logger.error(f"✗ Could not find configuration file: {config_file}")
            logger.error(f"  Tried paths: {config_paths_to_try}")
            results[config_file] = {
                "path": None,
                "valid": False,
                "error": "File not found"
            }
    
    return results

def check_directories():
    """Check required directories."""
    required_dirs = [
        "src/utils",
        "src/data",
        "src/training",
        "src/evaluation",
        "scripts/google_drive",
        "data/raw",
        "data/processed",
        "config",
        "models",
        "logs",
        "results",
        "visualizations"
    ]
    
    results = {}
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            logger.info(f"✓ Directory exists: {directory}")
            results[directory] = True
        else:
            logger.warning(f"✗ Directory missing: {directory}")
            results[directory] = False
    
    return results

def check_imports():
    """Check critical imports."""
    imports_to_check = [
        ("scripts.google_drive.google_drive_manager", "Google Drive Manager (scripts)"),
        ("src.utils.google_drive_manager", "Google Drive Manager (src)"),
        ("src.utils.drive_api_utils", "Drive API Utils"),
        ("src.data.process_datasets", "Data Processor")
    ]
    
    results = {}
    
    for module_path, description in imports_to_check:
        try:
            module = __import__(module_path, fromlist=["*"])
            logger.info(f"✓ Successfully imported {description}")
            results[module_path] = True
        except ImportError as e:
            logger.warning(f"✗ Failed to import {description}: {e}")
            results[module_path] = False
    
    return results

def check_credentials():
    """Check if credentials file exists and is valid."""
    credentials_path = "credentials.json"
    
    if not os.path.exists(credentials_path):
        logger.warning(f"✗ Credentials file not found: {credentials_path}")
        return {
            "exists": False,
            "valid": False,
            "hf_token": False
        }
    
    results = {
        "exists": True,
        "valid": False,
        "hf_token": False
    }
    
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        
        results["valid"] = True
        logger.info(f"✓ Credentials file is valid JSON")
        
        # Check for HF token
        hf_token = None
        
        if "huggingface" in credentials and "token" in credentials["huggingface"]:
            hf_token = credentials["huggingface"]["token"]
        elif "hf_token" in credentials:
            hf_token = credentials["hf_token"]
        elif "api_keys" in credentials and "huggingface" in credentials["api_keys"]:
            hf_token = credentials["api_keys"]["huggingface"]
        
        if hf_token:
            results["hf_token"] = True
            logger.info(f"✓ Found Hugging Face token in credentials")
        else:
            logger.warning(f"✗ No Hugging Face token found in credentials")
    
    except Exception as e:
        logger.error(f"✗ Error reading credentials file: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Verify paths and configurations")
    parser.add_argument("--fix", action="store_true", help="Try to fix issues automatically")
    args = parser.parse_args()
    
    logger.info("Verifying project setup...")
    
    # Add project root to path
    project_root = add_project_root()
    
    # Check directories
    logger.info("\nChecking required directories...")
    dir_results = check_directories()
    
    if args.fix:
        for directory, exists in dir_results.items():
            if not exists:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
    
    # Check configuration files
    logger.info("\nChecking configuration files...")
    config_results = check_config_files()
    
    # Check imports
    logger.info("\nChecking critical imports...")
    import_results = check_imports()
    
    # Check credentials
    logger.info("\nChecking credentials...")
    credentials_results = check_credentials()
    
    # Print summary
    logger.info("\n=== SUMMARY ===")
    
    missing_dirs = [d for d, exists in dir_results.items() if not exists]
    if missing_dirs:
        logger.warning(f"Missing directories: {', '.join(missing_dirs)}")
    else:
        logger.info("All required directories exist")
    
    missing_configs = [c for c, info in config_results.items() if info["path"] is None]
    if missing_configs:
        logger.warning(f"Missing config files: {', '.join(missing_configs)}")
    else:
        logger.info("All configuration files found")
    
    failed_imports = [i for i, success in import_results.items() if not success]
    if failed_imports:
        logger.warning(f"Failed imports: {', '.join(failed_imports)}")
    else:
        logger.info("All critical imports successful")
    
    if not credentials_results["exists"]:
        logger.warning("Credentials file missing")
    elif not credentials_results["valid"]:
        logger.warning("Credentials file is invalid")
    elif not credentials_results["hf_token"]:
        logger.warning("No Hugging Face token found in credentials")
    else:
        logger.info("Credentials are properly set up")
    
    # Final assessment
    if (not missing_dirs and 
        not missing_configs and 
        not failed_imports and 
        credentials_results["exists"] and 
        credentials_results["valid"]):
        logger.info("\n✓ Project is set up correctly!")
    else:
        logger.warning("\n✗ Some issues were found with the project setup")
        logger.info("Run paperspace_install.sh or scripts/fix_paperspace_imports.py to fix them")

if __name__ == "__main__":
    main() 