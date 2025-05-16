#!/usr/bin/env python3
"""
Create Training Symlinks

This script creates necessary symlinks for training files in the notebooks directory
to fix the 'file not found' errors in the training scripts.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_symlinks():
    """Create necessary symlinks for training files."""
    # Get project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up two levels from utilities to project root
    
    # Define source files and their target locations
    symlinks = [
        # Training files
        (project_root / "src" / "training" / "train.py", 
         Path("/notebooks") / "scripts" / "src" / "training" / "train.py"),
        (project_root / "src" / "training" / "trainer.py", 
         Path("/notebooks") / "scripts" / "src" / "training" / "trainer.py"),
        # Google Drive setup
        (project_root / "scripts" / "google_drive" / "setup_google_drive.py", 
         Path("/notebooks") / "scripts" / "setup_google_drive.py"),
        # Dataset preparation
        (project_root / "scripts" / "datasets" / "prepare_datasets_for_training.py", 
         Path("/notebooks") / "scripts" / "prepare_datasets_for_training.py"),
        # Feature extractor
        (project_root / "scripts" / "utilities" / "ensure_feature_extractor.py", 
         Path("/notebooks") / "scripts" / "ensure_feature_extractor.py"),
    ]
    
    # Create symlinks
    for source, target in symlinks:
        if not source.exists():
            logger.warning(f"Source file {source} does not exist. Skipping.")
            continue
            
        # Create parent directory if it doesn't exist
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink if it doesn't exist
        if not target.exists():
            try:
                os.symlink(source, target)
                logger.info(f"Created symlink: {source} -> {target}")
            except Exception as e:
                logger.error(f"Failed to create symlink {source} -> {target}: {e}")
        else:
            logger.info(f"Symlink already exists: {target}")

def main():
    """Main function."""
    # Check if we're in Paperspace
    if not Path("/notebooks").exists():
        logger.info("Not running in Paperspace. No need to create symlinks.")
        return
        
    logger.info("Creating training symlinks...")
    create_symlinks()
    logger.info("Done creating symlinks.")

if __name__ == "__main__":
    main()