#!/bin/bash
# cleanup.sh - Script to clean up redundant files and local data after Drive syncing
# This script helps maintain a clean workspace by removing cached and redundant files

set -e

# Process command line arguments
LOCAL_DATA_CLEANUP=false
ARCHIVE_OLD_SCRIPTS=false
CLEAN_CACHE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean-local-data)
      LOCAL_DATA_CLEANUP=true
      shift
      ;;
    --archive-old-scripts)
      ARCHIVE_OLD_SCRIPTS=true
      shift
      ;;
    --no-cache-clean)
      CLEAN_CACHE=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create archive directory if needed
if [ "$ARCHIVE_OLD_SCRIPTS" = true ]; then
  mkdir -p scripts/archived
  echo "==== Archiving redundant scripts ===="
  
  # List of scripts that are redundant with newer implementations
  REDUNDANT_SCRIPTS=(
    "setup_drive.py"     # Replaced by fix_drive_auth.py and google_drive_manager.py
    "sync_model.sh"      # Replaced by sync_to_drive.py
  )
  
  for script in "${REDUNDANT_SCRIPTS[@]}"; do
    if [ -f "scripts/$script" ]; then
      echo "Archiving $script"
      mv "scripts/$script" "scripts/archived/"
    fi
  done
  
  echo "Redundant scripts have been moved to scripts/archived/"
fi

# Clean up Python cache files
if [ "$CLEAN_CACHE" = true ]; then
  echo "==== Cleaning Python cache files ===="
  
  # Remove all __pycache__ directories
  find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  
  # Remove .pyc files
  find . -name "*.pyc" -delete
  
  # Remove .DS_Store files (macOS)
  find . -name ".DS_Store" -delete
  
  echo "Python cache files have been removed"
fi

# Clean up local data if requested
if [ "$LOCAL_DATA_CLEANUP" = true ]; then
  echo "==== Cleaning local data directories ===="
  
  # Ask for confirmation before proceeding
  read -p "This will remove local data files that should already be synced to Drive. Continue? (y/n) " -n 1 -r
  echo
  
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Clean up data directories (but keep directory structure)
    if [ -d "data/processed" ]; then
      echo "Removing processed datasets..."
      rm -rf data/processed/*
      echo "Kept 'data/processed' directory structure"
    fi
    
    # Clean up checkpoints
    if [ -d "models/checkpoints" ]; then
      echo "Removing model checkpoints..."
      rm -rf models/checkpoints/*
      echo "Kept 'models/checkpoints' directory structure"
    fi
    
    if [ -d "text_models/checkpoints" ]; then
      echo "Removing text model checkpoints..."
      rm -rf text_models/checkpoints/*
      echo "Kept 'text_models/checkpoints' directory structure"
    fi
    
    # Clean up log files but keep directory
    if [ -d "logs" ]; then
      echo "Removing log files..."
      find logs -type f -name "*.log" -delete
      echo "Kept 'logs' directory structure"
    fi
    
    echo "Local data cleanup completed"
  else
    echo "Local data cleanup canceled"
  fi
fi

echo "==== Cleanup completed! ====" 