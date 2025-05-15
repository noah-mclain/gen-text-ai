#!/bin/bash
# process_and_sync.sh - Process datasets with streaming and sync to Google Drive
# This script demonstrates the optimal workflow for minimizing memory and disk usage

# Set your datasets here or pass them as command line arguments
DATASETS=${@:-"code_alpaca mbpp humaneval codesearchnet_all codeparrot instruct_code"}

echo "🚀 Starting efficient dataset processing workflow"
echo "📊 Datasets to process: $DATASETS"
echo "💾 Using streaming mode to save memory"
echo "🔄 Will sync to Google Drive and delete local copies"

# Process datasets with streaming mode and skip local storage
python src/main_api.py \
  --mode process \
  --streaming \
  --no_cache \
  --skip_local_storage \
  --use_drive \
  --drive_base_dir DeepseekCoder \
  --datasets $DATASETS

echo "✅ Processing complete!"
echo ""
echo "To use these processed datasets for training, run:"
echo "python src/main_api.py --mode train --use_drive --drive_base_dir DeepseekCoder" 