#!/usr/bin/env python3
"""
Copy Feature Extractor

This script copies the feature_extractor.py file from the local repository to the
Paperspace environment to fix the 'feature extractor not found' error.

Run this script before starting the training process to ensure the feature extractor
is available in the correct location.
"""

import os
import sys
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get project root
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up two levels from utilities to project root

# Check if we're in Paperspace
in_paperspace = Path("/notebooks").exists()

def copy_feature_extractor():
    """Copy feature_extractor.py to the correct location in Paperspace."""
    # Source file in the local repository
    source_file = project_root / "src" / "data" / "processors" / "feature_extractor.py"
    
    if not source_file.exists():
        logger.error(f"Source feature extractor file not found at {source_file}")
        return False
    
    # Target locations
    target_locations = [
        # Main location in Paperspace
        Path("/notebooks") / "src" / "data" / "processors" / "feature_extractor.py",
        # Backup location in project root
        project_root / "src" / "data" / "processors" / "feature_extractor.py"
    ]
    
    success = True
    for target_file in target_locations:
        # Create parent directory if it doesn't exist
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy the file
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied feature extractor to {target_file}")
        except Exception as e:
            logger.error(f"Failed to copy feature extractor to {target_file}: {e}")
            success = False
    
    return success

def main():
    """Main function to copy the feature extractor."""
    logger.info("Starting feature extractor copy...")
    
    if not in_paperspace:
        logger.info("Not running in Paperspace. No need to copy feature extractor.")
        return 0
    
    # Copy feature extractor
    success = copy_feature_extractor()
    
    if success:
        logger.info("✅ Feature extractor copied successfully.")
        return 0
    else:
        logger.warning("⚠️ Feature extractor could not be copied.")
        return 1

if __name__ == "__main__":
    sys.exit(main())