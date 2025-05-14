#!/bin/bash

# Process text datasets with improved settings
# This script processes text-based datasets with proper error handling and dependencies

# Set bash to exit on error and show error trace
set -e

# Set up environment
export PYTHONPATH=.

# Create data directory structure
mkdir -p data/processed
PROCESSED_DIR=$(pwd)/data/processed
echo "Output directory: $PROCESSED_DIR"

# Check for required dependencies
echo "==== Installing Required Dependencies ===="
pip install --quiet zstandard langid nltk

# Download necessary NLTK resources 
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    print('Successfully downloaded NLTK punkt')
except Exception as e:
    print(f'Could not download NLTK data: {e}')
"

# Verify HF_TOKEN is available
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set. Some datasets might be inaccessible."
    echo "You can set it with: export HF_TOKEN=your_huggingface_token"
else
    echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Process the datasets with streaming mode for memory efficiency and detailed logging
# We also use no_cache to avoid disk space issues
echo "==== Processing Datasets ===="
LOG_FILE="logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

python -m src.data.process_datasets \
    --config config/dataset_config_text.json \
    --streaming \
    --no_cache 2>&1 | tee "$LOG_FILE"

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "✅ Dataset processing completed successfully!"
else
    echo "⚠️ Dataset processing encountered issues. Check the log for details."
fi

# Verify the output directory
echo "==== Checking Processed Datasets ===="

# Check if any processed datasets exist
DATASETS_COUNT=$(find "$PROCESSED_DIR" -type d -name "*_processed" | wc -l)
if [ "$DATASETS_COUNT" -eq 0 ]; then
    echo "⚠️ Warning: No processed datasets found in $PROCESSED_DIR"
    echo "This could indicate processing failures or incorrect output paths."
else
    echo "Found $DATASETS_COUNT processed dataset directories:"
    find "$PROCESSED_DIR" -type d -name "*_processed" | while read -r dir; do
        BASE_NAME=$(basename "$dir")
        echo "  - $BASE_NAME"
        # Check if the dataset contains any data
        if [ -f "$dir/dataset_info.json" ]; then
            echo "    ✓ Dataset appears to be valid (dataset_info.json exists)"
        else
            echo "    ⚠️ Dataset may be incomplete or empty (no dataset_info.json)"
        fi
    done
fi

echo "==== Processing Completed at $(date) ====" 