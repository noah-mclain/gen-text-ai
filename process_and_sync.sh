#!/bin/bash
# Script to process datasets and sync them to Google Drive

echo "🚀 Starting efficient dataset processing workflow"
echo "💾 Using streaming mode to save memory"
echo "🔄 Will sync to Google Drive and delete local copies"

# Define datasets as an array for better handling
DATASETS=("code_alpaca" "mbpp" "humaneval" "codesearchnet_all" "codeparrot" "instruct_code")

echo "📊 Processing datasets: ${DATASETS[*]}"

# Build the command with proper argument handling
CMD="python src/main_api.py --mode process --streaming --no_cache --skip_local_storage --dataset_config config/dataset_config.json --use_drive --drive_base_dir DeepseekCoder"

# Add each dataset as a separate argument for better compatibility
for dataset in "${DATASETS[@]}"; do
    CMD="$CMD --datasets $dataset"
done

echo "Running command: $CMD"
# Execute the command
eval $CMD

echo "✅ Processing complete!"
echo ""
echo "To use these processed datasets for training, run:"
echo "python src/main_api.py --mode train --use_drive --drive_base_dir DeepseekCoder" 