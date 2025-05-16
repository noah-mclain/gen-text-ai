#!/usr/bin/env python3
"""
Fix Feature Extractor Path

This script ensures the feature extractor module is properly found and loaded
by fixing import paths and creating necessary directories and symlinks.
"""

import os
import sys
import logging
from pathlib import Path
import importlib
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_feature_extractor_path():
    """Fix the feature extractor path and ensure it can be imported."""
    # Get project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up two levels from utilities to project root
    
    # Add project root to path
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        logger.info(f"Added {project_root} to sys.path")
    
    # Check if we're in Paperspace
    in_paperspace = Path("/notebooks").exists()
    if in_paperspace:
        # Add notebooks to path
        if "/notebooks" not in sys.path:
            sys.path.append("/notebooks")
            logger.info("Added /notebooks to sys.path")
        
        # Create necessary directories
        data_dir = Path("/notebooks/data/processed/features/deepseek_coder")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {data_dir}")
    
    # Ensure the feature extractor module exists
    feature_extractor_dir = project_root / "src" / "data" / "processors"
    feature_extractor_file = feature_extractor_dir / "feature_extractor.py"
    
    if not feature_extractor_file.exists():
        logger.error(f"Feature extractor file not found at {feature_extractor_file}")
        # Try to find it elsewhere in the project
        found_files = list(project_root.glob("**/feature_extractor.py"))
        if found_files:
            logger.info(f"Found feature extractor at {found_files[0]}")
            # Create the directory if it doesn't exist
            feature_extractor_dir.mkdir(parents=True, exist_ok=True)
            # Copy the file
            shutil.copy(found_files[0], feature_extractor_file)
            logger.info(f"Copied feature extractor to {feature_extractor_file}")
        else:
            logger.error("Could not find feature_extractor.py anywhere in the project")
            return False
    
    # Try to import the feature extractor
    try:
        # Clear any previous imports
        if "src.data.processors.feature_extractor" in sys.modules:
            del sys.modules["src.data.processors.feature_extractor"]
        
        # Try to import
        importlib.import_module("src.data.processors.feature_extractor")
        logger.info("Successfully imported feature extractor module")
        return True
    except ImportError as e:
        logger.error(f"Failed to import feature extractor: {e}")
        return False

def main():
    """Main function."""
    logger.info("Fixing feature extractor path...")
    success = fix_feature_extractor_path()
    
    if success:
        logger.info("Feature extractor path fixed successfully.")
    else:
        logger.error("Failed to fix feature extractor path.")
        sys.exit(1)

if __name__ == "__main__":
    main()