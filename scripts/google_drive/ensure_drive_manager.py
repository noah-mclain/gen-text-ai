#!/usr/bin/env python3
"""
Ensure Google Drive Manager

This script ensures that the proper Google Drive manager implementation
is used by setting up Python's import paths and verifying dependencies.

Usage:
    python -m scripts.google_drive.ensure_drive_manager
"""

import os
import sys
import importlib
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_import_paths():
    """Ensure proper import paths are set up for Google Drive manager."""
    # Get project paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Ensure project root is in path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to path: {project_root}")
    
    # Ensure src directory is in path
    src_path = os.path.join(project_root, 'src')
    if os.path.exists(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)
        logger.info(f"Added src directory to path: {src_path}")
    
    # For Paperspace, also add /notebooks path
    notebooks_path = "/notebooks"
    if os.path.exists(notebooks_path) and notebooks_path not in sys.path:
        sys.path.insert(0, notebooks_path)
        logger.info(f"Added notebooks directory to path: {notebooks_path}")
    
    # Check for different src directory in Paperspace
    paperspace_src = os.path.join(notebooks_path, 'src')
    if os.path.exists(paperspace_src) and paperspace_src not in sys.path:
        sys.path.insert(0, paperspace_src)
        logger.info(f"Added Paperspace src directory to path: {paperspace_src}")
    
    return True

def ensure_google_api_available():
    """Ensure Google API libraries are installed."""
    try:
        # Try importing required libraries
        import google.oauth2.credentials
        import google_auth_oauthlib.flow
        import googleapiclient.discovery
        logger.info("Google API libraries are already installed")
        return True
    except ImportError:
        # Libraries not installed, try to install them
        logger.warning("Google API libraries not available. Attempting to install...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib"])
            logger.info("Successfully installed Google API libraries")
            return True
        except Exception as e:
            logger.error(f"Failed to install Google API libraries: {e}")
            logger.info("Please manually install the required packages with:")
            logger.info("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False

def check_drive_manager_implementation():
    """Check which Google Drive manager implementation is being used."""
    # Fix import paths first
    fix_import_paths()
    
    # Try to import from different locations and check implementation
    implementations = []
    
    # Implementation 1: src/utils/google_drive_manager.py
    try:
        from src.utils.google_drive_manager import GOOGLE_API_AVAILABLE
        implementations.append(("src.utils.google_drive_manager", GOOGLE_API_AVAILABLE))
    except (ImportError, AttributeError):
        logger.warning("Could not import from src.utils.google_drive_manager")
    
    # Implementation 2: scripts/src/utils/google_drive_manager.py
    try:
        # Clear any previous import to avoid conflicts
        if "scripts.src.utils.google_drive_manager" in sys.modules:
            del sys.modules["scripts.src.utils.google_drive_manager"]
            
        from scripts.src.utils.google_drive_manager import GOOGLE_API_AVAILABLE
        implementations.append(("scripts.src.utils.google_drive_manager", GOOGLE_API_AVAILABLE))
    except (ImportError, AttributeError):
        logger.warning("Could not import from scripts.src.utils.google_drive_manager")
    
    # Check if we have any implementations
    if not implementations:
        logger.error("No Google Drive manager implementation found")
        return False
    
    # Check if we have a valid implementation (with Google API)
    valid_implementations = [imp for imp, has_api in implementations if has_api]
    if valid_implementations:
        logger.info(f"Found valid implementation(s): {', '.join(valid_implementations)}")
        return True
    else:
        logger.warning(f"Found implementation(s) but none have Google API available: {implementations}")
        # Try to install Google API
        if ensure_google_api_available():
            logger.info("Google API libraries installed, please try again")
        return False

def copy_implementation_if_needed():
    """Copy implementation file if it doesn't exist in the right location."""
    # Define source and target paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    source_paths = [
        os.path.join(project_root, 'src', 'utils', 'google_drive_manager.py'),
        os.path.join(project_root, 'scripts', 'src', 'utils', 'google_drive_manager.py'),
        os.path.join('/notebooks', 'src', 'utils', 'google_drive_manager.py')
    ]
    
    target_paths = [
        os.path.join(project_root, 'src', 'utils', 'google_drive_manager.py'),
        os.path.join('/notebooks', 'src', 'utils', 'google_drive_manager.py')
    ]
    
    # Find a valid source
    valid_source = None
    for path in source_paths:
        if os.path.exists(path):
            # Check if it's a valid implementation
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if 'DriveManager' in content and 'GOOGLE_API_AVAILABLE' in content:
                        valid_source = path
                        logger.info(f"Found valid implementation at: {path}")
                        break
            except Exception as e:
                logger.warning(f"Error reading file {path}: {e}")
    
    if not valid_source:
        logger.error("No valid implementation found to copy")
        return False
    
    # Copy to target locations
    success = False
    for target in target_paths:
        if target != valid_source:  # Don't copy to itself
            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir):
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    logger.info(f"Created directory: {target_dir}")
                except Exception as e:
                    logger.warning(f"Failed to create directory {target_dir}: {e}")
                    continue
            
            try:
                shutil.copy2(valid_source, target)
                logger.info(f"Copied implementation from {valid_source} to {target}")
                success = True
            except Exception as e:
                logger.warning(f"Failed to copy to {target}: {e}")
    
    return success

def main():
    """Main function to ensure Google Drive manager implementation."""
    print("\nEnsuring proper Google Drive manager implementation...")
    
    # Fix import paths
    fix_import_paths()
    
    # Check if Google API is available
    api_available = ensure_google_api_available()
    if not api_available:
        print("\nGoogle API libraries are not available. Please install them manually.")
        return 1
    
    # Check drive manager implementation
    implementation_ok = check_drive_manager_implementation()
    if not implementation_ok:
        print("\nNo valid Google Drive manager implementation found.")
        print("Attempting to copy implementation...")
        copy_success = copy_implementation_if_needed()
        if copy_success:
            print("\nSuccessfully copied Google Drive manager implementation.")
            # Check again after copying
            implementation_ok = check_drive_manager_implementation()
        else:
            print("\nFailed to copy Google Drive manager implementation.")
    
    if implementation_ok:
        print("\n✅ Google Drive manager is properly set up!")
        return 0
    else:
        print("\n❌ Failed to set up Google Drive manager properly.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 