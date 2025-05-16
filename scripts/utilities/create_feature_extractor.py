#!/usr/bin/env python3
"""
Create Feature Extractor in Paperspace

This is a minimal script that doesn't import any external libraries and just
creates the feature_extractor.py file in the Paperspace environment.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_feature_extractor():
    """Create the feature_extractor.py file in the Paperspace environment."""
    # Define target path in Paperspace
    notebooks_path = Path("/notebooks")
    if not notebooks_path.exists():
        logger.info("Not running in Paperspace. No need to create feature extractor.")
        return True
    
    target_file = notebooks_path / "src" / "data" / "processors" / "feature_extractor.py"
    target_dir = target_file.parent
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Try to find the source file
    source_paths = [
        Path(__file__).parent.parent.parent / "src" / "data" / "processors" / "feature_extractor.py",
        notebooks_path / "gen-text-ai" / "src" / "data" / "processors" / "feature_extractor.py"
    ]
    
    source_content = None
    for source_file in source_paths:
        if source_file.exists():
            try:
                with open(source_file, 'r') as f:
                    source_content = f.read()
                logger.info(f"Found source feature extractor at {source_file}")
                break
            except Exception as e:
                logger.warning(f"Error reading source file {source_file}: {e}")
    
    # If we didn't find a source file, use the fallback
    if not source_content:
        logger.info("Using fallback feature extractor content")
        
        # Read from the content file if available
        content_file = Path(__file__).parent / "feature_extractor_content.py"
        if content_file.exists():
            try:
                with open(content_file, 'r') as f:
                    source_content = f.read()
                logger.info(f"Read feature extractor content from {content_file}")
            except Exception as e:
                logger.warning(f"Error reading content file {content_file}: {e}")
                return False
        else:
            logger.error(f"No fallback content file found at {content_file}")
            return False
    
    # Write the content to the target file
    try:
        with open(target_file, 'w') as f:
            f.write(source_content)
        logger.info(f"Successfully created feature extractor at {target_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to create feature extractor: {e}")
        return False

def main():
    """Main function to create the feature extractor."""
    logger.info("Starting feature extractor creation...")
    
    # Create feature extractor
    success = create_feature_extractor()
    
    if success:
        logger.info("✅ Feature extractor created successfully.")
        return 0
    else:
        logger.warning("⚠️ Feature extractor could not be created.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 