#!/bin/bash
# Script to organize files within subdirectories for better structure

echo "===== GEN-TEXT-AI SUBDIRECTORY ORGANIZATION ====="
echo "This script will organize files within each major directory"
echo "into more specific subdirectories based on purpose."
echo

# Parse command line arguments
DRY_RUN=""

if [ "$1" == "--dry-run" ]; then
  DRY_RUN="--dry-run"
  echo "Performing DRY RUN - no files will be moved"
else
  echo "WARNING: This will move files to subdirectories within major directories."
  echo "Run with --dry-run to see what changes would be made without moving files."
  echo
  echo "Do you want to continue? (y/n)"
  read -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
  fi
fi

# Run the organization script
echo "Starting subdirectory organization..."
python organize_subdirectories.py $DRY_RUN

# Check if the organization was successful
if [ $? -ne 0 ]; then
  echo "Subdirectory organization encountered errors."
  exit 1
fi

# Update symbolic links if this was a real run
if [ -z "$DRY_RUN" ]; then
  echo "Updating symbolic links..."
  
  # Manually update the symbolic links
  # Remove any existing symlinks
  if [ -L "train_a6000_optimized.sh" ]; then
    rm train_a6000_optimized.sh
  fi
  
  if [ -L "train_deepseek_coder.sh" ]; then
    rm train_deepseek_coder.sh
  fi
  
  if [ -L "fix_dataset_features.sh" ]; then
    rm fix_dataset_features.sh
  fi
  
  # Create new symlinks
  if [ -f "scripts/training/train_a6000_optimized.sh" ]; then
    ln -s scripts/training/train_a6000_optimized.sh train_a6000_optimized.sh
    echo "Created symlink: train_a6000_optimized.sh -> scripts/training/train_a6000_optimized.sh"
  fi
  
  if [ -f "scripts/training/train_deepseek_coder.sh" ]; then
    ln -s scripts/training/train_deepseek_coder.sh train_deepseek_coder.sh
    echo "Created symlink: train_deepseek_coder.sh -> scripts/training/train_deepseek_coder.sh"
  fi
  
  if [ -f "scripts/datasets/fix_dataset_features.sh" ]; then
    ln -s scripts/datasets/fix_dataset_features.sh fix_dataset_features.sh
    echo "Created symlink: fix_dataset_features.sh -> scripts/datasets/fix_dataset_features.sh"
  fi
fi

echo
echo "===== SUBDIRECTORY ORGANIZATION COMPLETE ====="

if [ -z "$DRY_RUN" ]; then
  echo "Files have been organized into appropriate subdirectories."
  echo "Symbolic links have been updated for commonly used scripts."
  echo "The codebase is now organized in a more structured way!"
else
  echo "This was a dry run. Run without --dry-run to make actual changes."
fi 