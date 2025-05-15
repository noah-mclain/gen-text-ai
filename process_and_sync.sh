#!/bin/bash
# Script to process datasets and sync them to Google Drive

echo "🚀 Starting efficient dataset processing workflow"
echo "💾 Using streaming mode to save memory"
echo "🔄 Will sync to Google Drive and delete local copies"

# Process datasets using the process_datasets script with all needed options
# This will:
# 1. Process datasets in streaming mode to save memory
# 2. Save processed datasets to a temporary directory
# 3. Sync them to Google Drive
# 4. Delete the local copies to save disk space

python src/main_api.py \
  --mode process \
  --streaming \
  --no_cache \
  --skip_local_storage \
  --use_drive \
  --drive_base_dir DeepseekCoder

echo "✅ Processing complete!"
echo ""
echo "To use these processed datasets for training, run:"
echo "python src/main_api.py --mode train --use_drive --drive_base_dir DeepseekCoder" 