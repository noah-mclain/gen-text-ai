#!/bin/bash
# Script to organize the codebase by moving files to appropriate directories

echo "===== GEN-TEXT-AI CODEBASE ORGANIZATION ====="
echo "This script will organize files in the root directory"
echo "by moving them to appropriate subdirectories."
echo

# Parse command line arguments
DRY_RUN=""
SKIP_IMPORTS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN="--dry-run"
      shift
      ;;
    --skip-imports)
      SKIP_IMPORTS="--skip-imports"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dry-run] [--skip-imports]"
      exit 1
      ;;
  esac
done

# Display mode information
if [ -n "$DRY_RUN" ]; then
  echo "Performing DRY RUN - no files will be moved"
else
  echo "WARNING: This will move files to their appropriate directories."
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

if [ -n "$SKIP_IMPORTS" ]; then
  echo "Import statements will NOT be updated."
fi

# Run the cleanup script
echo "Starting codebase organization..."

# Check for the reorganized path first
if [ -f "scripts/utilities/codebase_cleanup.py" ]; then
  CLEANUP_SCRIPT="scripts/utilities/codebase_cleanup.py"
elif [ -f "scripts/codebase_cleanup.py" ]; then
  CLEANUP_SCRIPT="scripts/codebase_cleanup.py"
else
  echo "Error: Could not find codebase_cleanup.py in either scripts/ or scripts/utilities/"
  exit 1
fi

python $CLEANUP_SCRIPT $DRY_RUN $SKIP_IMPORTS

# Check if the cleanup was successful
if [ $? -ne 0 ]; then
  echo "Codebase organization encountered errors."
  exit 1
fi

# Create symbolic links for commonly used scripts
if [ -z "$DRY_RUN" ]; then
  echo "Creating symbolic links for commonly used scripts..."
  
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
  
  # Create new symlinks with updated paths
  if [ -f "scripts/training/train_a6000_optimized.sh" ]; then
    ln -s scripts/training/train_a6000_optimized.sh train_a6000_optimized.sh
    echo "Created symlink: train_a6000_optimized.sh -> scripts/training/train_a6000_optimized.sh"
  elif [ -f "scripts/train_a6000_optimized.sh" ]; then
    ln -s scripts/train_a6000_optimized.sh train_a6000_optimized.sh
    echo "Created symlink: train_a6000_optimized.sh -> scripts/train_a6000_optimized.sh"
  fi
  
  if [ -f "scripts/training/train_deepseek_coder.sh" ]; then
    ln -s scripts/training/train_deepseek_coder.sh train_deepseek_coder.sh
    echo "Created symlink: train_deepseek_coder.sh -> scripts/training/train_deepseek_coder.sh"
  elif [ -f "scripts/train_deepseek_coder.sh" ]; then
    ln -s scripts/train_deepseek_coder.sh train_deepseek_coder.sh
    echo "Created symlink: train_deepseek_coder.sh -> scripts/train_deepseek_coder.sh"
  fi
  
  if [ -f "scripts/datasets/fix_dataset_features.sh" ]; then
    ln -s scripts/datasets/fix_dataset_features.sh fix_dataset_features.sh
    echo "Created symlink: fix_dataset_features.sh -> scripts/datasets/fix_dataset_features.sh"
  elif [ -f "scripts/fix_dataset_features.sh" ]; then
    ln -s scripts/fix_dataset_features.sh fix_dataset_features.sh
    echo "Created symlink: fix_dataset_features.sh -> scripts/fix_dataset_features.sh"
  fi
fi

echo
echo "===== CODEBASE ORGANIZATION COMPLETE ====="

if [ -z "$DRY_RUN" ]; then
  echo "Files have been moved to their appropriate directories."
  echo "Symbolic links have been created for commonly used scripts."
  echo "The codebase is now better organized!"
else
  echo "This was a dry run. Run without --dry-run to make actual changes."
fi 