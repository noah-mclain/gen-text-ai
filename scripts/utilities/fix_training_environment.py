#!/usr/bin/env python3
"""
Fix Training Environment

This script addresses all the issues identified in the training pipeline:
1. Creates necessary symlinks for training files
2. Fixes feature extractor path issues
3. Ensures proper dataset loading with Arrow format
4. Sets up Google Drive integration correctly

Run this script before starting the training process to ensure everything is set up correctly.
"""

import os
import sys
import logging
from pathlib import Path
import importlib
import shutil
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def create_symlinks():
    """Create necessary symlinks for training files."""
    if not in_paperspace:
        logger.info("Not running in Paperspace. No need to create symlinks.")
        return True
        
    logger.info("Creating training symlinks...")
    
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
    
    success = True
    # Create symlinks
    for source, target in symlinks:
        if not source.exists():
            logger.warning(f"Source file {source} does not exist. Skipping.")
            success = False
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
                success = False
        else:
            logger.info(f"Symlink already exists: {target}")
    
    return success

def fix_feature_extractor():
    """Fix feature extractor path and ensure it can be imported."""
    logger.info("Fixing feature extractor path...")
    
    # Ensure the feature extractor module exists
    feature_extractor_dir = project_root / "src" / "data" / "processors"
    feature_extractor_file = feature_extractor_dir / "feature_extractor.py"
    
    if not feature_extractor_file.exists():
        logger.warning(f"Feature extractor file not found at {feature_extractor_file}")
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
            # Try to use the copy_feature_extractor script
            try:
                copy_script_path = current_dir / "copy_feature_extractor.py"
                if copy_script_path.exists():
                    logger.info("Attempting to run copy_feature_extractor.py script...")
                    result = subprocess.run([sys.executable, str(copy_script_path)], 
                                          capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        logger.info("Successfully copied feature extractor using copy script")
                        return True
                    else:
                        logger.warning(f"Copy script failed: {result.stderr}")
                else:
                    logger.warning("Copy feature extractor script not found")
            except Exception as e:
                logger.error(f"Error running copy script: {e}")
            
            # Try to create the feature extractor directly in Paperspace
            try:
                ensure_script_path = current_dir / "ensure_paperspace_feature_extractor.py"
                if ensure_script_path.exists():
                    logger.info("Attempting to create feature extractor directly in Paperspace...")
                    result = subprocess.run([sys.executable, str(ensure_script_path)], 
                                          capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        logger.info("Successfully created feature extractor in Paperspace")
                        return True
                    else:
                        logger.warning(f"Feature extractor creation failed: {result.stderr}")
                else:
                    logger.warning("Ensure feature extractor script not found")
            except Exception as e:
                logger.error(f"Error creating feature extractor: {e}")
            
            # If we get here, we couldn't find or copy the feature extractor
            logger.error("Could not find feature_extractor.py anywhere in the project")
            return False
    
    # Create necessary directories for feature extraction
    features_dir = project_root / "data" / "processed" / "features" / "deepseek_coder"
    features_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created features directory: {features_dir}")
    
    # If in Paperspace, also create the directory there
    if in_paperspace:
        # Create feature extractor directory
        paperspace_feature_extractor_dir = Path("/notebooks") / "src" / "data" / "processors"
        paperspace_feature_extractor_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created Paperspace feature extractor directory: {paperspace_feature_extractor_dir}")
        
        # Create features directory
        paperspace_features_dir = Path("/notebooks") / "data" / "processed" / "features" / "deepseek_coder"
        paperspace_features_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created Paperspace features directory: {paperspace_features_dir}")
    
    return True

def fix_dataset_loading():
    """Fix dataset loading to properly support Arrow format."""
    logger.info("Fixing dataset loading for Arrow format...")
    
    try:
        # Check if pyarrow is installed
        import importlib.util
        if importlib.util.find_spec("pyarrow") is None:
            logger.warning("PyArrow not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
            logger.info("PyArrow installed successfully.")
        
        # Import necessary modules
        from datasets import Dataset, load_from_disk
        import pyarrow as pa
        
        logger.info("Successfully imported datasets and pyarrow modules")
        
        # Create necessary directories
        features_dir = project_root / "data" / "processed" / "features" / "deepseek_coder"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a small test dataset in Arrow format to verify functionality
        test_data = {
            "text": ["This is a test dataset to verify Arrow format loading."],
            "metadata": [{"source": "test"}]
        }
        
        # Create test directory
        test_dir = features_dir / "test_dataset"
        
        # Create dataset and save it
        test_dataset = Dataset.from_dict(test_data)
        test_dataset.save_to_disk(test_dir)
        logger.info(f"Created test dataset at {test_dir}")
        
        # Try loading the test dataset
        loaded_dataset = load_from_disk(test_dir)
        logger.info(f"Successfully loaded test dataset with {len(loaded_dataset)} examples")
        
        # Patch dataset loading functions to use Arrow format
        # Find all Python files that might contain dataset loading code
        for py_file in project_root.glob("**/*.py"):
            if "venv" in str(py_file) or ".git" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, "r") as f:
                    content = f.read()
                    
                # Check if file contains JSON dataset loading code
                if "json.load" in content and "datasets" in content and "save_to_disk" not in content:
                    logger.info(f"Found potential JSON dataset loading in {py_file}")
                    
                    # Replace JSON loading with Arrow loading where appropriate
                    new_content = content.replace(
                        "json.load(f)", 
                        "load_from_disk(path) if os.path.isdir(path) else json.load(f)"
                    )
                    
                    if new_content != content:
                        with open(py_file, "w") as f:
                            f.write(new_content)
                        logger.info(f"Updated {py_file} to support Arrow format")
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")
        
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required packages: pip install datasets pyarrow")
        return False
    except Exception as e:
        logger.error(f"Error fixing dataset loading: {e}")
        return False

def setup_google_drive():
    """Set up Google Drive integration."""
    logger.info("Setting up Google Drive integration...")
    
    # Check if the setup script exists
    setup_script = project_root / "scripts" / "google_drive" / "setup_google_drive.py"
    if not setup_script.exists():
        logger.error(f"Google Drive setup script not found at {setup_script}")
        return False
    
    # Run the setup script
    try:
        result = subprocess.run([sys.executable, str(setup_script)], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.warning(f"Google Drive setup failed with error: {result.stderr}")
            logger.warning("Will proceed without Google Drive integration.")
            return False
        else:
            logger.info("Google Drive setup completed successfully.")
            return True
    except Exception as e:
        logger.error(f"Error running Google Drive setup: {e}")
        return False

def main():
    """Main function to fix all training environment issues."""
    logger.info("Starting training environment fix...")
    
    # Create symlinks
    symlinks_success = create_symlinks()
    if symlinks_success:
        logger.info("✅ Training symlinks created successfully.")
    else:
        logger.warning("⚠️ Some training symlinks could not be created.")
    
    # Fix feature extractor
    feature_extractor_success = fix_feature_extractor()
    if feature_extractor_success:
        logger.info("✅ Feature extractor path fixed successfully.")
    else:
        logger.warning("⚠️ Feature extractor path could not be fixed.")
    
    # Fix dataset loading
    dataset_loading_success = fix_dataset_loading()
    if dataset_loading_success:
        logger.info("✅ Dataset loading fixed successfully.")
    else:
        logger.warning("⚠️ Dataset loading could not be fixed.")
    
    # Setup Google Drive
    google_drive_success = setup_google_drive()
    if google_drive_success:
        logger.info("✅ Google Drive integration set up successfully.")
    else:
        logger.warning("⚠️ Google Drive integration could not be set up.")
    
    # Overall success
    if symlinks_success and feature_extractor_success and dataset_loading_success and google_drive_success:
        logger.info("✅ All training environment issues fixed successfully!")
        return 0
    else:
        logger.warning("⚠️ Some training environment issues could not be fixed.")
        logger.info("Please check the logs for details and try running the training script anyway.")
        return 1

if __name__ == "__main__":
    sys.exit(main())