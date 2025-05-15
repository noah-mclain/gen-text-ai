#!/bin/bash
# Script to process datasets and sync them to Google Drive

echo "🚀 Starting efficient dataset processing workflow"
echo "💾 Using streaming mode to save memory"
echo "🔄 Will sync to Google Drive and delete local copies"

# Create a visible temporary directory in the workspace
TEMP_DIR="./temp_datasets_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"
echo "📁 Using temporary directory: $TEMP_DIR"

# Process datasets using the process_datasets script with all needed options
# This will:
# 1. Process datasets in streaming mode to save memory
# 2. Save processed datasets to a visible temporary directory
# 3. Sync them to Google Drive
# 4. Delete the local copies to save disk space

python src/main_api.py \
  --mode process \
  --streaming \
  --no_cache \
  --skip_local_storage \
  --use_drive \
  --drive_base_dir "datasets" \
  --temp_dir "$TEMP_DIR"

echo "✅ Processing complete!"
echo ""
echo "🔍 Check Google Drive folder 'datasets' for your processed datasets"
echo "To use these processed datasets for training, run:"
echo "python src/main_api.py --mode train --use_drive --drive_base_dir datasets"

# Clean up temporary directory if needed
echo "🧹 Cleaning up temporary directory..."
rm -rf "$TEMP_DIR" 