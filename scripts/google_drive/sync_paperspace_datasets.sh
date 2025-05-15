#!/bin/bash
# Sync Paperspace datasets to Google Drive
# This script finds all processed datasets in Paperspace and syncs them to Google Drive

set -e  # Exit on error

echo "===== SYNCING PAPERSPACE DATASETS TO GOOGLE DRIVE ====="

# Parse command line arguments
USE_PREPROCESSED=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --use-preprocessed)
      USE_PREPROCESSED=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--use-preprocessed]"
      exit 1
      ;;
  esac
done

# Add project root to Python path
export PYTHONPATH=.

# Check for datasets
PAPERSPACE_DIR="/notebooks/data/processed"
LOCAL_DATA_DIR="data/processed"
PROCESSED_COUNT=0

if [ -d "$PAPERSPACE_DIR" ]; then
    echo "Checking Paperspace directory: $PAPERSPACE_DIR"
    PROCESSED_COUNT=$(find "$PAPERSPACE_DIR" -type d -name "*_processed" | wc -l)
    echo "Found $PROCESSED_COUNT processed datasets in Paperspace directory"
else
    echo "WARNING: Paperspace directory $PAPERSPACE_DIR not found"
fi

if [ -d "$LOCAL_DATA_DIR" ]; then
    echo "Checking local directory: $LOCAL_DATA_DIR"
    LOCAL_COUNT=$(find "$LOCAL_DATA_DIR" -type d -name "*_processed" | wc -l)
    echo "Found $LOCAL_COUNT processed datasets in local directory"
    PROCESSED_COUNT=$((PROCESSED_COUNT + LOCAL_COUNT))
else
    echo "WARNING: Local directory $LOCAL_DATA_DIR not found"
fi

if [ "$PROCESSED_COUNT" -eq 0 ]; then
    echo "ERROR: No processed datasets found to sync"
    exit 1
fi

echo "Ready to sync $PROCESSED_COUNT datasets to Google Drive"

# Automatically sync using our Python utility with the force flag
echo "Running sync utility..."
if [ "$USE_PREPROCESSED" = true ]; then
    echo "Using 'preprocessed' folder for compatibility with training scripts"
    python scripts/google_drive/sync_processed_datasets.py --force --use-preprocessed
    DRIVE_FOLDER="preprocessed"
else
    echo "Using 'data/processed' folder for compatibility with Google Drive structure"
    python scripts/google_drive/sync_processed_datasets.py --force
    DRIVE_FOLDER="data/processed"
fi

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Datasets successfully synced to Google Drive"
    echo "They are available in the DeepseekCoder/$DRIVE_FOLDER/ folder"
else
    echo "❌ Error syncing datasets to Google Drive"
    echo "Try running with verbose output for debugging:"
    if [ "$USE_PREPROCESSED" = true ]; then
        echo "python scripts/google_drive/sync_processed_datasets.py --force --use-preprocessed"
    else
        echo "python scripts/google_drive/sync_processed_datasets.py --force"
    fi
    exit 1
fi

echo "===== SYNC COMPLETE =====" 