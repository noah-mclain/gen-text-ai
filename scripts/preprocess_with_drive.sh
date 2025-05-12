#!/bin/bash
# preprocess_with_drive.sh - Script to preprocess datasets with Google Drive sync
# This script shows how to use the new Google Drive integration options

set -e

# Set default parameters
DRIVE_BASE_DIR="DeepseekCoder"
USE_RCLONE=true
STREAMING=true
NO_CACHE=true
SKIP_LOCAL=true
DATASETS="the_stack_filtered codesearchnet_all code_alpaca humaneval mbpp codeparrot instruct_code"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --use-api)
      USE_RCLONE=false
      shift
      ;;
    --keep-local)
      SKIP_LOCAL=false
      shift
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create the logs directory if it doesn't exist
mkdir -p logs
echo "Created logs directory"

# Set up command with proper options
if [ "$USE_RCLONE" = true ]; then
  DRIVE_OPTS="--use_drive"
else
  DRIVE_OPTS="--use_drive_api"
fi

# Add other options
if [ "$SKIP_LOCAL" = true ]; then
  DRIVE_OPTS="$DRIVE_OPTS --skip_local_storage"
fi

# Print info
echo "==== Preprocessing datasets with Google Drive integration ===="
echo "Drive base directory: $DRIVE_BASE_DIR"
echo "Using rclone: $USE_RCLONE"
echo "Skip local storage: $SKIP_LOCAL"
echo "Datasets: $DATASETS"

# Run the preprocessing command
echo "Starting preprocessing..."
python main_api.py \
  --mode process \
  --datasets $DATASETS \
  --streaming \
  --no_cache \
  --dataset_config config/dataset_config.json \
  $DRIVE_OPTS \
  --drive_base_dir $DRIVE_BASE_DIR \
  2>&1 | tee logs/preprocessing_$(date +%Y%m%d_%H%M%S).log

echo "==== Preprocessing complete! ===="
echo "Check the logs directory for details." 