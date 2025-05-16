#!/usr/bin/env python3
"""
Copy Feature Extractor to Paperspace

This script simply copies the feature_extractor.py file to the Paperspace environment
without importing any libraries that might cause syntax errors.
"""

import os
import sys
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_feature_extractor():
    """Copy the feature_extractor.py file to the Paperspace environment."""
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up two levels from utilities to project root
    
    source_file = project_root / "src" / "data" / "processors" / "feature_extractor.py"
    if not source_file.exists():
        source_file = Path("/notebooks/gen-text-ai/src/data/processors/feature_extractor.py")
        if not source_file.exists():
            logger.error(f"Source feature extractor file not found at {source_file}")
            return False
    
    # Define target path in Paperspace
    notebooks_path = Path("/notebooks")
    if not notebooks_path.exists():
        logger.info("Not running in Paperspace. No need to copy feature extractor.")
        return True
    
    target_file = notebooks_path / "src" / "data" / "processors" / "feature_extractor.py"
    target_dir = target_file.parent
    
    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the file
    try:
        shutil.copy2(source_file, target_file)
        logger.info(f"Successfully copied feature extractor from {source_file} to {target_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy feature extractor: {e}")
        return False

def write_feature_extractor_directly():
    """Create the feature_extractor.py file directly in Paperspace."""
    # Define target path in Paperspace
    notebooks_path = Path("/notebooks")
    if not notebooks_path.exists():
        logger.info("Not running in Paperspace. No need to create feature extractor.")
        return True
    
    target_file = notebooks_path / "src" / "data" / "processors" / "feature_extractor.py"
    target_dir = target_file.parent
    
    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the content from the src file
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    source_file = project_root / "src" / "data" / "processors" / "feature_extractor.py"
    
    try:
        if source_file.exists():
            with open(source_file, 'r') as f:
                content = f.read()
        else:
            # Fallback to hardcoded content from the repository
            with open(Path(__file__).parent / "feature_extractor_content.py", 'r') as f:
                content = f.read()
        
        # Write the content to the target file
        with open(target_file, 'w') as f:
            f.write(content)
            
        logger.info(f"Successfully created feature extractor at {target_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create feature extractor: {e}")
        return False

def main():
    """Main function to copy the feature extractor."""
    logger.info("Starting feature extractor copy...")
    
    # First try to copy the file
    success = copy_feature_extractor()
    
    # If copying fails, try to write directly
    if not success:
        logger.warning("Copying failed, trying to write feature extractor directly...")
        success = write_feature_extractor_directly()
    
    if success:
        logger.info("✅ Feature extractor setup successfully.")
        return 0
    else:
        logger.warning("⚠️ Feature extractor could not be copied or created.")
        return 1

if __name__ == "__main__":
    sys.exit(main())