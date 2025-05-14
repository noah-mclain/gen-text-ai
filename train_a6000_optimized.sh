#!/bin/bash
# Fully optimized training script for A6000 GPU with 48GB VRAM
# This script is configured for maximum performance with multi-language support

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create function to calculate expected training time
calculate_training_time() {
    # Inputs to function
    local num_epochs=$1
    local num_datasets=$2
    local model_name=$(grep -o '"model_name_or_path": "[^"]*"' config/training_config.json | cut -d'"' -f4)
    local batch_size=$(grep -o '"per_device_train_batch_size": [0-9]*' config/training_config.json | grep -o '[0-9]*')
    local grad_accum=$(grep -o '"gradient_accumulation_steps": [0-9]*' config/training_config.json | grep -o '[0-9]*')

    # Default values if not found
    if [ -z "$batch_size" ]; then batch_size=1; fi
    if [ -z "$grad_accum" ]; then grad_accum=16; fi
    
    # Base time calculation - minutes per epoch per dataset
    local base_time_per_epoch=30  # Base minutes per epoch for a single dataset
    
    # Adjust for model size - larger models take longer
    local model_scale=1.0
    if [[ "$model_name" == *"deepseek-coder-6.7b"* ]]; then
        model_scale=1.5
    elif [[ "$model_name" == *"deepseek-coder-33b"* ]]; then
        model_scale=4.0
    fi
    
    # Adjust for effective batch size (batch_size * grad_accum)
    local batch_factor=$(echo "scale=2; 16 / ($batch_size * $grad_accum)" | bc)
    if (( $(echo "$batch_factor < 0.5" | bc -l) )); then batch_factor=0.5; fi
    if (( $(echo "$batch_factor > 2.0" | bc -l) )); then batch_factor=2.0; fi
    
    # Calculate total hours
    local total_minutes=$(echo "$base_time_per_epoch * $num_epochs * $num_datasets * $model_scale * $batch_factor" | bc)
    local total_hours=$(echo "scale=1; $total_minutes / 60" | bc)
    
    # Add buffer time (15%)
    total_hours=$(echo "scale=1; $total_hours * 1.15" | bc)
    
    # Round up to nearest integer
    total_hours=$(echo "scale=0; $total_hours+0.5" | bc)
    if [ "$total_hours" -lt 1 ]; then total_hours=1; fi
    
    echo "$total_hours"
}

# Read number of epochs from training config
NUM_EPOCHS=$(grep -o '"num_train_epochs": [0-9]*' config/training_config.json | grep -o '[0-9]*')
if [ -z "$NUM_EPOCHS" ]; then
  NUM_EPOCHS=1  # Default to 1 if not found
fi
echo "Detected $NUM_EPOCHS epochs in training configuration"

# Count number of enabled datasets
NUM_DATASETS=$(grep -o '"dataset_weights": {' -A 20 config/training_config.json | grep -o ":" | wc -l)
echo "Training on approximately $NUM_DATASETS datasets for $NUM_EPOCHS epochs"

# Calculate estimated training time
MAX_HOURS=$(calculate_training_time $NUM_EPOCHS $NUM_DATASETS)
echo "Estimated training time: $MAX_HOURS hours"

# Calculate expected completion time
START_TIME=$(date +%s)
END_TIME=$((START_TIME + MAX_HOURS * 3600))
COMPLETION_TIME=$(date -r $END_TIME "+%Y-%m-%d %H:%M:%S")
echo "Expected completion time: $COMPLETION_TIME"

# Override MAX_HOURS if provided as a command-line argument
if [ "$1" != "" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  MAX_HOURS=$1
  echo "Using command-line specified training time: $MAX_HOURS hours"
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible and you won't be able to push to Hugging Face Hub."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
else
  echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Fix Unsloth import order warning
export PYTHONPATH=$PYTHONPATH:.

# Make directories
mkdir -p logs
mkdir -p data/processed

# Set up Google Drive authentication first
echo "Setting up Google Drive authentication..."
python scripts/setup_google_drive.py
if [ $? -ne 0 ]; then
    echo "Google Drive authentication setup failed. Will proceed without Drive integration."
    USE_DRIVE_FLAG=""
    SKIP_DRIVE="--skip_drive"
else
    echo "Google Drive authentication successful. Will use Drive for storage."
    USE_DRIVE_FLAG="--use_drive --drive_base_dir DeepseekCoder"
    SKIP_DRIVE=""
fi

# Clean any cached files for better memory management
echo "Cleaning cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock

# Pre-download datasets with retry mechanism (fallback option)
echo "Pre-warming dataset cache for faster loading..."
python -c "from datasets import load_dataset; load_dataset('code-search-net/code_search_net', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('sahil2801/CodeAlpaca-20k', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('mbpp', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('codeparrot/codeparrot-clean', split='train', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('openai/openai_humaneval', split='test', streaming=True, trust_remote_code=True)"
python -c "from datasets import load_dataset; load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', streaming=True, trust_remote_code=True)"

# ===== DATASET PREPARATION FLOW =====
# 1. Check Google Drive for preprocessed datasets
# 2. Download available preprocessed datasets
# 3. Process only datasets not found on Drive
# 4. Train on all datasets

# Check if datasets are already available on Google Drive
echo "===== CHECKING FOR PRE-PROCESSED DATASETS ====="
if [ "${SKIP_DRIVE:-False}" = "False" ]; then
    echo "Google Drive integration is ENABLED. Looking for pre-processed datasets on Drive..."
else
    echo "Google Drive integration is DISABLED (SKIP_DRIVE=$SKIP_DRIVE). Using local datasets only."
fi

# Get list of datasets that need processing vs ones available on Drive
echo "Running dataset availability check..."
DATASET_STATUS=$(python -c "
from src.utils.drive_dataset_checker import prepare_datasets
import json
import os
import sys

config_path = 'config/dataset_config.json'
skip_drive = ${SKIP_DRIVE:-False}

# This gets datasets available locally or on Drive, and those still needed
available, needed, download_time = prepare_datasets(
    config_path, 
    output_dir='data/processed',
    skip_drive=skip_drive
)

print('AVAILABLE=' + ','.join(available), file=sys.stdout)
print('NEEDED=' + ','.join(needed), file=sys.stdout)
print('DOWNLOAD_TIME=' + str(download_time), file=sys.stdout)
")

# Parse results
AVAILABLE_DATASETS=$(echo "$DATASET_STATUS" | grep "AVAILABLE=" | cut -d'=' -f2)
DATASETS_TO_PROCESS=$(echo "$DATASET_STATUS" | grep "NEEDED=" | cut -d'=' -f2)
DOWNLOAD_TIME=$(echo "$DATASET_STATUS" | grep "DOWNLOAD_TIME=" | cut -d'=' -f2)

# Show status
echo "===== DATASET STATUS ====="
echo "Already available datasets: $AVAILABLE_DATASETS"
echo "Datasets that need processing: $DATASETS_TO_PROCESS"
echo "Download time from Drive: $DOWNLOAD_TIME seconds"

# Only process datasets if there are any that need processing
if [ -n "$DATASETS_TO_PROCESS" ]; then
    echo "===== PROCESSING DATASETS ====="
    echo "Processing datasets: $DATASETS_TO_PROCESS"
    
    python main_api.py \
        --mode process \
        --datasets $DATASETS_TO_PROCESS \
        --streaming \
        --no_cache \
        --dataset_config config/dataset_config.json \
        $USE_DRIVE_FLAG \
        2>&1 | tee logs/dataset_processing_$(date +%Y%m%d_%H%M%S).log
        
    if [ $? -ne 0 ]; then
        echo "⚠️ Dataset processing encountered errors. Training may use incomplete data."
    else
        echo "✅ Dataset processing completed successfully."
    fi
else
    echo "✅ All datasets already available. Skipping dataset processing step."
fi

# Update config to ensure DeepSpeed is disabled
echo "===== UPDATING CONFIG FOR STANDARD TRAINING ====="
python -c "
import json
import sys
try:
    with open('config/training_config.json', 'r') as f:
        config = json.load(f)
    
    if 'training' not in config:
        config['training'] = {}
    
    # Ensure DeepSpeed is disabled in config
    if 'use_deepspeed' in config['training']:
        config['training']['use_deepspeed'] = False
        print('Updated config: DeepSpeed disabled')
    
    # Also update any other DeepSpeed-related settings
    if 'deepspeed_config' in config['training']:
        del config['training']['deepspeed_config']
        print('Removed DeepSpeed config path from settings')
    
    with open('config/training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print('Config file updated successfully')
except Exception as e:
    print(f'Error updating config: {e}', file=sys.stderr)
"

echo "===== STARTING TRAINING ====="
TRAINING_START_TIME=$(date +%s)

# Train with direct module call (bypassing main_api.py)
echo "Starting training with direct module call (avoids argument mismatch)..."
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    $USE_DRIVE_FLAG \
    --push_to_hub \
    --no_deepspeed \
    2>&1 | tee logs/train_a6000_optimized_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_STATUS=$?

# Calculate actual training time
TRAINING_END_TIME=$(date +%s)
ACTUAL_TRAINING_TIME=$((TRAINING_END_TIME - TRAINING_START_TIME))
ACTUAL_HOURS=$(echo "scale=2; $ACTUAL_TRAINING_TIME / 3600" | bc)

# Report completion
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Training completed successfully!"
  echo "Actual training time: $ACTUAL_HOURS hours (estimated: $MAX_HOURS hours)"
  
  # Create completion marker file with timestamp for tracking
  echo "Training completed at $(date)" > logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  echo "Actual training time: $ACTUAL_HOURS hours" >> logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  echo "Estimated training time: $MAX_HOURS hours" >> logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
  
  # Sync results to Drive if available
  if [ -n "$USE_DRIVE_FLAG" ]; then
    echo "Syncing results to Google Drive..."
    python scripts/sync_to_drive.py --sync-all
  fi
else
  echo "Training failed with exit code $EXIT_STATUS. Check the logs for details."
  echo "Training ran for $ACTUAL_HOURS hours before failing (estimated: $MAX_HOURS hours)"
fi

# Calculate training statistics if log file exists
LOG_FILE=$(find logs -name "train_a6000_optimized_*.log" -type f -exec ls -t {} \; | head -1)
if [ -f "$LOG_FILE" ]; then
  echo "Analyzing training log for statistics..."
  echo "Training duration: $(grep -o "Training runtime.*" $LOG_FILE | tail -1)"
  echo "Samples processed: $(grep -o "trained on [0-9]* samples" $LOG_FILE | tail -1)"
  echo "Final loss: $(grep -o "loss=.*" $LOG_FILE | tail -1)"
  
  # Calculate samples per second
  SAMPLES=$(grep -o "trained on [0-9]* samples" $LOG_FILE | tail -1 | grep -o "[0-9]*")
  if [ -n "$SAMPLES" ] && [ "$ACTUAL_TRAINING_TIME" -gt 0 ]; then
    SAMPLES_PER_SEC=$(echo "scale=2; $SAMPLES / $ACTUAL_TRAINING_TIME" | bc)
    echo "Processing speed: $SAMPLES_PER_SEC samples/second"
  fi
  
  # Add training speed to completion file
  echo "Processing speed: $SAMPLES_PER_SEC samples/second" >> logs/training_complete_$(date +%Y%m%d_%H%M%S).txt
fi

echo "Training process complete at $(date)" 