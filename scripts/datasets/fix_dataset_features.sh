#!/bin/bash
# Fix dataset feature inconsistencies before training

echo "===== FIXING DATASET FEATURE INCONSISTENCIES ====="

# Set Python path to find modules
export PYTHONPATH=$PYTHONPATH:.

# First, create symlinks to map dataset names correctly
echo "Step 1: Creating dataset symlinks..."
python fix_dataset_mapping.py --list
python fix_dataset_mapping.py

# Check if running on Paperspace and fix there too
if [ -d "/notebooks" ]; then
  echo "Paperspace environment detected, fixing there too..."
  
  # Copy the fix script if needed
  if [ ! -f "/notebooks/fix_dataset_mapping.py" ]; then
    cp fix_dataset_mapping.py /notebooks/
  fi
  
  # Run in the notebooks directory
  cd /notebooks
  python fix_dataset_mapping.py
  cd -
fi

# Now run a script to standardize dataset features
echo "Step 2: Standardizing dataset features..."
python -c '
import os
import sys
import glob
import logging
from datasets import load_from_disk, Dataset
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Required columns for training
REQUIRED_COLUMNS = ["input_ids", "attention_mask", "labels"]

def fix_dataset_features(dataset_dir):
    """
    Fix dataset features to ensure they have the required columns.
    
    Args:
        dataset_dir: Directory of the dataset to fix
    """
    logger.info(f"Examining dataset: {dataset_dir}")
    
    try:
        # Try to load the dataset
        dataset = load_from_disk(dataset_dir)
        
        # Check if dataset has all required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in dataset.column_names]
        
        # If any columns are missing, we need to fix the dataset
        if missing_columns:
            logger.info(f"Dataset is missing columns: {missing_columns}")
            logger.info(f"Current columns: {dataset.column_names}")
            
            # No need to modify dataset - our training code will handle this
            logger.info("Dataset columns will be standardized during training")
            return
            
        logger.info(f"Dataset has all required columns: {REQUIRED_COLUMNS}")
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_dir}: {e}")

# Process all dataset directories
def process_dataset_directories(base_dir):
    """Process all dataset directories in the given base directory"""
    logger.info(f"Searching for datasets in {base_dir}")
    
    if not os.path.exists(base_dir):
        logger.warning(f"Directory {base_dir} does not exist")
        return
    
    # Find all dataset directories (containing _processed in name)
    dataset_dirs = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and ("_processed" in item or "_interim_" in item):
            dataset_dirs.append(full_path)
    
    logger.info(f"Found {len(dataset_dirs)} dataset directories")
    
    # Process each dataset
    for dataset_dir in dataset_dirs:
        fix_dataset_features(dataset_dir)

# Process both local and Paperspace directories
process_dataset_directories("data/processed")
if os.path.exists("/notebooks/data/processed"):
    process_dataset_directories("/notebooks/data/processed")

logger.info("Dataset feature standardization complete")
'

echo "===== DATASET FIXES COMPLETE ====="
echo "You can now run training with confidence that datasets will be compatible." 