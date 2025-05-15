#!/bin/bash
# Script to process datasets and sync them to Google Drive

echo "🚀 Starting efficient dataset processing workflow"
echo "💾 Using streaming mode to save memory"
echo "🔄 Will sync to Google Drive and delete local copies"

# Process specific datasets with streaming mode and skip local storage
# These match the enabled datasets in your config file
python src/main_api.py \
  --mode process \
  --datasets code_alpaca mbpp humaneval codesearchnet_all codeparrot instruct_code \
  --streaming \
  --no_cache \
  --skip_local_storage \
  --dataset_config config/dataset_config.json \
  --use_drive \
  --drive_base_dir DeepseekCoder

echo "✅ Processing complete!"
echo ""
echo "To use these processed datasets for training, run:"
echo "python src/main_api.py --mode train --use_drive --drive_base_dir DeepseekCoder" 