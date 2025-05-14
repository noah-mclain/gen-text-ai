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
    
    # Also check for specific Google API libraries
    google_api_libraries = [
        ("google.auth", "google-auth"),
        ("googleapiclient", "google-api-python-client"),
        ("google_auth_oauthlib", "google-auth-oauthlib")
    ]
    
    results = {}
    missing_google_libs = []
    
    # Check Google API libraries
    for module_name, package_name in google_api_libraries:
        try:
            __import__(module_name)
        except ImportError:
            missing_google_libs.append(package_name)
    
    if missing_google_libs:
        logger.warning(f"Google API libraries not installed. Install with: pip install {' '.join(missing_google_libs)}")
    
    # Check main imports
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
    """Check if credentials file exists and is valid, and if HF token is set."""
    credentials_path = "credentials.json"
    
    results = {
        "exists": False,
        "valid": False,
        "hf_token": False,
        "hf_token_source": None
    }
    
    # First check if HF_TOKEN is in environment variables
    if os.environ.get("HF_TOKEN"):
        logger.info(f"✓ Found HF_TOKEN in environment variables")
        results["hf_token"] = True
        results["hf_token_source"] = "environment"
    
    # Then check credentials file
    if not os.path.exists(credentials_path):
        logger.warning(f"✗ Credentials file not found: {credentials_path}")
        
        # If HF token is already found in environment, still consider it a "success"
        if results["hf_token"]:
            results["exists"] = True  # We're considering the absence of credentials file OK if HF_TOKEN env var exists
            return results
            
        return results
    
    results["exists"] = True
    
    try:
        with open(credentials_path, 'r') as f:
            credentials = json.load(f)
        
        results["valid"] = True
        logger.info(f"✓ Credentials file is valid JSON")
        
        # Only check for HF token in credentials if not already found in environment
        if not results["hf_token"]:
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
                results["hf_token_source"] = "credentials"
                logger.info(f"✓ Found Hugging Face token in credentials")
            else:
                logger.info(f"No Hugging Face token found in credentials (using environment variable)")
    
    except Exception as e:
        logger.error(f"✗ Error reading credentials file: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Verify paths and configurations")
    parser.add_argument("--fix", action="store_true", help="Try to fix issues automatically")
    parser.add_argument("--install-deps", action="store_true", help="Install missing dependencies")
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
        if credentials_results["hf_token"]:
            logger.info("Credentials file missing, but HF_TOKEN is set in environment")
        else:
            logger.warning("Credentials file missing and no HF_TOKEN in environment")
    elif not credentials_results["valid"]:
        if credentials_results["hf_token"]:
            logger.info("Credentials file is invalid, but HF_TOKEN is set in environment")
        else:
            logger.warning("Credentials file is invalid and no HF_TOKEN in environment")
    elif not credentials_results["hf_token"]:
        logger.warning("No Hugging Face token found in credentials or environment")
    else:
        if credentials_results["hf_token_source"] == "environment":
            logger.info("HF_TOKEN is set in environment variables")
        else:
            logger.info("Credentials are properly set up with Hugging Face token")
    
    # Install dependencies if requested
    if args.install_deps:
        logger.info("\nInstalling missing dependencies...")
        try:
            import subprocess
            
            # Check for Google API libraries
            try:
                import google.auth
                import googleapiclient
                import google_auth_oauthlib
            except ImportError:
                logger.info("Installing Google API libraries...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "google-api-python-client",
                    "google-auth-httplib2",
                    "google-auth-oauthlib"
                ])
            
            # Check for other common dependencies
            try:
                import torch
            except ImportError:
                logger.info("Installing PyTorch...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "torch"
                ])
            
            logger.info("Dependencies installation complete!")
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
    
    # Final assessment
    if (not missing_dirs and 
        not missing_configs and 
        not failed_imports and 
        (credentials_results["hf_token"] or 
         (credentials_results["exists"] and credentials_results["valid"]))):
        logger.info("\n✓ Project is set up correctly!")
    else:
        logger.warning("\n✗ Some issues were found with the project setup")
        logger.info("Run paperspace_install.sh or scripts/fix_paperspace_imports.py to fix them")
        
        if not args.install_deps:
            logger.info("You can also run 'python scripts/utilities/verify_paths.py --install-deps' to install missing dependencies")

if __name__ == "__main__":
    main() 