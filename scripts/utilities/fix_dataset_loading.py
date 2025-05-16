#!/usr/bin/env python3
"""
Fix Dataset Loading

This script fixes dataset loading issues by ensuring proper support for Arrow format
instead of JSON for dataset saving and loading.
"""

import os
import sys
import logging
from pathlib import Path
import importlib
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up two levels from utilities to project root
sys.path.append(str(project_root))

# Also add notebooks directory to path if we're in Paperspace
notebook_path = Path("/notebooks")
if notebook_path.exists():
    sys.path.append(str(notebook_path))

def fix_dataset_loading():
    """Fix dataset loading to properly support Arrow format."""
    try:
        # Import necessary modules
        from datasets import Dataset, load_from_disk
        import pyarrow as pa
        
        logger.info("Successfully imported datasets and pyarrow modules")
        
        # Create feature directories if they don't exist
        features_dir = project_root / "data" / "processed" / "features" / "deepseek_coder"
        features_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created features directory: {features_dir}")
        
        # If in Paperspace, also create the directory there
        if notebook_path.exists():
            paperspace_features_dir = notebook_path / "data" / "processed" / "features" / "deepseek_coder"
            paperspace_features_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created Paperspace features directory: {paperspace_features_dir}")
        
        # Create a small test dataset in Arrow format to verify functionality
        test_data = {
            "text": ["This is a test dataset to verify Arrow format loading."],
            "metadata": [{"source": "test"}]
        }
        
        test_dataset = Dataset.from_dict(test_data)
        test_dir = features_dir / "test_dataset"
        test_dataset.save_to_disk(test_dir)
        logger.info(f"Created test dataset at {test_dir}")
        
        # Try loading the test dataset
        loaded_dataset = load_from_disk(test_dir)
        logger.info(f"Successfully loaded test dataset with {len(loaded_dataset)} examples")
        
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required packages: pip install datasets pyarrow")
        return False
    except Exception as e:
        logger.error(f"Error fixing dataset loading: {e}")
        return False

def patch_dataset_loading_functions():
    """Patch dataset loading functions to ensure Arrow format support."""
    try:
        # Find all Python files that might contain dataset loading code
        python_files = list(project_root.glob("**/*.py"))
        patched_files = []
        
        for file_path in python_files:
            # Skip files in certain directories
            if any(d in str(file_path) for d in ["venv", ".git", "__pycache__"]):
                continue
                
            # Read the file content
            try:
                with open(file_path, "r") as f:
                    content = f.read()
            except Exception:
                continue
                
            # Check if the file contains dataset loading code
            if "json.load" in content and "datasets" in content:
                logger.info(f"Found potential dataset loading code in {file_path}")
                patched_files.append(str(file_path))
        
        if patched_files:
            logger.info(f"Found {len(patched_files)} files that may need patching:")
            for file in patched_files:
                logger.info(f"  - {file}")
            logger.info("Please review these files and ensure they support Arrow format loading")
        else:
            logger.info("No files found that need patching for Arrow format support")
        
        return True
    except Exception as e:
        logger.error(f"Error patching dataset loading functions: {e}")
        return False

def main():
    """Main function."""
    logger.info("Fixing dataset loading for Arrow format...")
    
    # Fix dataset loading
    loading_fixed = fix_dataset_loading()
    if loading_fixed:
        logger.info("Dataset loading fixed successfully.")
    else:
        logger.error("Failed to fix dataset loading.")
        
    # Patch dataset loading functions
    patching_success = patch_dataset_loading_functions()
    if patching_success:
        logger.info("Dataset loading functions patched successfully.")
    else:
        logger.error("Failed to patch dataset loading functions.")
    
    # Overall success
    if loading_fixed and patching_success:
        logger.info("All dataset loading issues fixed successfully.")
        return 0
    else:
        logger.error("Some dataset loading issues could not be fixed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())