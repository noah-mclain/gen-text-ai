#!/bin/bash
# Fully optimized training script for A6000 GPU with 48GB VRAM
# This script is configured for maximum performance with multi-language support

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set DeepSpeed environment variables
export DS_ACCELERATOR=cuda
export DS_OFFLOAD_PARAM=cpu
export DS_OFFLOAD_OPTIMIZER=cpu
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=ds_config_a6000.json

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible and you won't be able to push to Hugging Face Hub."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
else
  echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Fix Unsloth import order warning
export PYTHONPATH=$PYTHONPATH:.
# Create a wrapper for imports
cat > import_wrapper.py << EOL
from unsloth import FastLanguageModel
import transformers
import peft
print("✅ Imports properly ordered for optimal performance")
EOL
python import_wrapper.py

# Make directories
mkdir -p logs
mkdir -p data/processed

# Set up Google Drive authentication first
echo "Setting up Google Drive authentication..."
python scripts/setup_google_drive.py
if [ $? -ne 0 ]; then
    echo "Google Drive authentication setup failed. Will proceed without Drive integration."
    USE_DRIVE_FLAG=""
else
    echo "Google Drive authentication successful. Will use Drive for storage."
    USE_DRIVE_FLAG="--use_drive --drive_base_dir DeepseekCoder"
fi

# Clean any cached files for better memory management
echo "Cleaning cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock
rm -rf ~/.cache/huggingface/transformers/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/*/*.safetensors.tmp*

# Set max training time based on system time (default to 10 hours if not specified)
if [ -z "$MAX_HOURS" ]; then
  # Calculate hours until midnight - force interpretation as base 10
  CURRENT_HOUR=$(date +%H | sed 's/^0*//')
  # If CURRENT_HOUR is empty after removing leading zeros, it was midnight (00), so set to 0
  if [ -z "$CURRENT_HOUR" ]; then
    CURRENT_HOUR=0
  fi
  
  MAX_HOURS=$((24 - 10#$CURRENT_HOUR - 1))
  
  # Ensure we have at least 2 hours of training
  if [ $MAX_HOURS -lt 2 ]; then
    MAX_HOURS=10
  fi
  
  echo "Auto-calculated training time: $MAX_HOURS hours"
else
  echo "Using specified training time: $MAX_HOURS hours"
fi

# Pre-download datasets with retry mechanism
echo "Pre-downloading datasets to ensure availability..."
python -c "from datasets import load_dataset; load_dataset('code-search-net/code_search_net', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('mbpp', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('openai/openai_humaneval', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', streaming=True, trust_remote_code=True)"

# Process datasets first
echo "Processing datasets..."
python main_api.py \
    --mode process \
    --datasets the_stack_filtered codesearchnet_all code_alpaca humaneval mbpp codeparrot instruct_code \
    --streaming \
    --no_cache \
    --dataset_config config/dataset_config.json \
    2>&1 | tee logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log

# Train with direct module call (bypassing main_api.py)
echo "Starting training with direct module call (avoids argument mismatch)..."
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    $USE_DRIVE_FLAG \
    --push_to_hub \
    2>&1 | tee logs/train_a6000_optimized_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_STATUS=$?

# Report completion
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Training completed successfully!"
  
  # Create completion marker file with timestamp for tracking
  echo "Training completed at $(date)" > logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  
  # Sync results to Drive if available
  if [ -n "$USE_DRIVE_FLAG" ]; then
    echo "Syncing results to Google Drive..."
    python scripts/sync_to_drive.py --sync-all
  fi
else
  echo "Training failed with exit code $EXIT_STATUS. Check the logs for details."
fi

# Calculate training statistics if log file exists
LOG_FILE=$(find logs -name "train_a6000_optimized_*.log" -type f -exec ls -t {} \; | head -1)
if [ -f "$LOG_FILE" ]; then
  echo "Analyzing training log for statistics..."
  echo "Training duration: $(grep -o "Training runtime.*" $LOG_FILE | tail -1)"
  echo "Samples processed: $(grep -o "trained on [0-9]* samples" $LOG_FILE | tail -1)"
  echo "Final loss: $(grep -o "loss=.*" $LOG_FILE | tail -1)"
fi

echo "Training process complete at $(date)" 