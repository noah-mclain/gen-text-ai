#!/usr/bin/env python3
"""
Fix Paperspace Imports

This script helps set up the Paperspace environment to work with the
project's codebase by ensuring all the right files and paths are in place.

Run this script at the start of your Paperspace session:
python scripts/fix_paperspace_imports.py
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """
    Set up the Python path to include the project root.
    """
    # Get the path to the current script
    current_script = os.path.abspath(__file__)
    scripts_dir = os.path.dirname(current_script)
    project_root = os.path.dirname(scripts_dir)
    
    # Add project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added project root to Python path: {project_root}")
    
    # Set up environment variable for future Python processes
    os.environ["PYTHONPATH"] = f"{project_root}:" + os.environ.get("PYTHONPATH", "")
    logger.info(f"Set PYTHONPATH environment variable to include: {project_root}")
    
    return project_root

def ensure_directory_structure():
    """
    Ensure all necessary directories exist.
    """
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "results",
        "visualizations",
        "config",
        "src/utils",
        "src/data",
        "src/training",
        "src/evaluation",
        "scripts/google_drive",
        "scripts/src/utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def copy_google_drive_manager(project_root):
    """
    Make sure Google Drive manager implementation is available in all necessary locations.
    
    Args:
        project_root: Path to the project root directory
    """
    # Define source and target paths
    source_paths = [
        os.path.join(project_root, "src", "utils", "google_drive_manager.py"),
        os.path.join(project_root, "scripts", "google_drive", "google_drive_manager_impl.py")
    ]
    
    target_paths = [
        os.path.join(project_root, "src", "utils", "google_drive_manager.py"),
        os.path.join(project_root, "scripts", "src", "utils", "google_drive_manager.py"),
        os.path.join(project_root, "scripts", "google_drive", "google_drive_manager_impl.py")
    ]
    
    # Find the source file
    source_file = None
    for path in source_paths:
        if os.path.exists(path):
            source_file = path
            logger.info(f"Found Google Drive manager implementation: {source_file}")
            break
    
    if not source_file:
        logger.error("Could not find Google Drive manager implementation")
        return
    
    # Copy to all target locations
    for target_path in target_paths:
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            shutil.copy2(source_file, target_path)
            logger.info(f"Copied implementation to: {target_path}")
        except Exception as e:
            logger.error(f"Failed to copy to {target_path}: {e}")

def create_symbolic_links():
    """
    Create symbolic links to make imports work regardless of current directory.
    """
    links = [
        {"source": "src/utils", "target": "scripts/src/utils"},
        {"source": "src/data", "target": "scripts/src/data"}
    ]
    
    for link in links:
        source = os.path.abspath(link["source"])
        target = os.path.abspath(link["target"])
        
        # Create the target directory if it doesn't exist
        os.makedirs(os.path.dirname(target), exist_ok=True)
        
        # Create symbolic link if needed
        if not os.path.exists(target):
            try:
                # Use relative path for the link
                source_rel = os.path.relpath(source, os.path.dirname(target))
                os.symlink(source_rel, target, target_is_directory=True)
                logger.info(f"Created symbolic link: {target} -> {source_rel}")
            except OSError as e:
                logger.error(f"Failed to create symbolic link {target}: {e}")
                # If symlink fails, try copying files instead
                if os.path.isdir(source):
                    shutil.copytree(source, target, dirs_exist_ok=True)
                    logger.info(f"Copied directory contents from {source} to {target}")

def main():
    parser = argparse.ArgumentParser(description="Fix import issues in Paperspace environment")
    parser.add_argument("--root", type=str, default=None, 
                       help="Project root directory (default: auto-detect)")
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.root if args.root else setup_paths()
    
    # Ensure directories exist
    ensure_directory_structure()
    
    # Copy Google Drive manager implementation
    copy_google_drive_manager(project_root)
    
    # Create symbolic links
    try:
        create_symbolic_links()
    except Exception as e:
        logger.error(f"Error creating symbolic links: {e}")
    
    logger.info("Environment setup complete")
    logger.info("You can now run your code in the Paperspace environment")

if __name__ == "__main__":
    main() 