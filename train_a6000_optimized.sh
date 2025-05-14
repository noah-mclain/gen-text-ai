#!/bin/bash
# Fully optimized training script for A6000 GPU with 48GB VRAM
# This script is configured for maximum performance with multi-language support

# Make diagnose_deepspeed.sh executable
chmod +x scripts/diagnose_deepspeed.sh 2>/dev/null || true

# Function to verify DeepSpeed setup
verify_deepspeed_setup() {
    # Check if DeepSpeed is available
    if ! python -c "import deepspeed" 2>/dev/null; then
        echo "⚠️ DeepSpeed not found. Installing..."
        pip install deepspeed --no-cache-dir
    fi
    
    # Check environment variables
    if [ -z "$ACCELERATE_USE_DEEPSPEED" ] || [ -z "$ACCELERATE_DEEPSPEED_PLUGIN_TYPE" ] || [ -z "$HF_DS_CONFIG" ]; then
        echo "⚠️ Some DeepSpeed environment variables are not set. Running fix script..."
        python scripts/fix_deepspeed.py
    fi
    
    # Check if the config file exists
    if [ ! -f "$ACCELERATE_DEEPSPEED_CONFIG_FILE" ] && [ ! -f "$HF_DS_CONFIG" ]; then
        echo "⚠️ DeepSpeed config file not found. Creating one..."
        python scripts/fix_deepspeed.py
    fi
    
    echo "DeepSpeed setup verification complete."
}

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Read number of epochs from training config
NUM_EPOCHS=$(grep -o '"num_train_epochs": [0-9]*' config/training_config.json | grep -o '[0-9]*')
if [ -z "$NUM_EPOCHS" ]; then
  NUM_EPOCHS=1  # Default to 1 if not found
fi
echo "Detected $NUM_EPOCHS epochs in training configuration"

# Calculate default MAX_HOURS based on epochs (2 hours per epoch)
export MAX_HOURS=$((2 * NUM_EPOCHS))
echo "Setting default training time to $MAX_HOURS hours based on $NUM_EPOCHS epochs"

# Create Triton autotune directory to prevent df error
mkdir -p /root/.triton/autotune 2>/dev/null || true

# Set DeepSpeed environment variables
export DS_ACCELERATOR=cuda
export DS_OFFLOAD_PARAM=cpu
export DS_OFFLOAD_OPTIMIZER=cpu
export ACCELERATE_USE_DEEPSPEED=true

# Fix DeepSpeed configuration - ensure the config is valid and accessible
echo "===== SETTING UP DEEPSPEED CONFIGURATION ====="
# Run the DeepSpeed fix script
python scripts/fix_deepspeed.py

# Get the absolute path to DeepSpeed config 
DS_CONFIG_PATH=$(realpath ds_config_a6000.json)
if [ ! -f "$DS_CONFIG_PATH" ]; then
  echo "⚠️ DeepSpeed config not found at $DS_CONFIG_PATH. Creating default config."
  cat > "$DS_CONFIG_PATH" << EOL
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "train_batch_size": 32,
  "train_micro_batch_size_per_gpu": 4,
  "wall_clock_breakdown": false
}
EOL
fi

# Set all necessary environment variables for DeepSpeed
export ACCELERATE_DEEPSPEED_CONFIG_FILE="$DS_CONFIG_PATH"
# Explicitly set plugin type to fix 'NoneType' object has no attribute 'hf_ds_config' error
export ACCELERATE_DEEPSPEED_PLUGIN_TYPE=deepspeed
# Make sure HF_DS_CONFIG is set for transformers to recognize DeepSpeed config
export HF_DS_CONFIG="$DS_CONFIG_PATH"
# Optional but recommended for better performance
export TRANSFORMERS_ZeRO_2_FORCE_INVALIDATE_CHECKPOINT=1

echo "DeepSpeed config at: $ACCELERATE_DEEPSPEED_CONFIG_FILE"
echo "DeepSpeed plugin type: $ACCELERATE_DEEPSPEED_PLUGIN_TYPE"
echo "HF_DS_CONFIG: $HF_DS_CONFIG"

# Make a local backup copy of the DeepSpeed config for debugging
mkdir -p logs/deepspeed
cp "$DS_CONFIG_PATH" "logs/deepspeed/ds_config_backup_$(date +%Y%m%d_%H%M%S).json"

# Copy DeepSpeed config to Paperspace notebooks directory if running there
if [ -d "/notebooks" ]; then
  echo "Paperspace environment detected. Copying DeepSpeed config to /notebooks directory..."
  cp "$DS_CONFIG_PATH" /notebooks/ds_config_a6000.json
  echo "Config copied successfully to /notebooks/ds_config_a6000.json"
  
  # Also create directory for models config
  mkdir -p /notebooks/models
  cp "$DS_CONFIG_PATH" /notebooks/models/ds_config.json
  echo "Config copied to /notebooks/models/ds_config.json for model initialization"
fi

# Test DeepSpeed configuration if script exists
if [ -f "scripts/test_deepspeed_config.py" ]; then
  echo "===== TESTING DEEPSPEED CONFIGURATION ====="
  python scripts/test_deepspeed_config.py || echo "⚠️ DeepSpeed configuration test completed with warnings (continuing anyway)"
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
    SKIP_DRIVE="--skip_drive"
else
    echo "Google Drive authentication successful. Will use Drive for storage."
    USE_DRIVE_FLAG="--use_drive --drive_base_dir DeepseekCoder"
    SKIP_DRIVE=""
fi

# Clean any cached files for better memory management
echo "Cleaning cache directories..."
rm -rf ~/.cache/huggingface/datasets/downloads/completed.lock
rm -rf ~/.cache/huggingface/transformers/models--deepseek-ai--deepseek-coder-6.7b-base/snapshots/*/*.safetensors.tmp*

# Override MAX_HOURS if provided as a command-line argument
if [ "$1" != "" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  MAX_HOURS=$1
  echo "Using command-line specified training time: $MAX_HOURS hours"
else
  # Count number of enabled datasets
  NUM_DATASETS=$(grep -o '"dataset_weights": {' -A 20 config/training_config.json | grep -o ":" | wc -l)
  echo "Training on approximately $NUM_DATASETS datasets for $NUM_EPOCHS epochs"
  echo "Estimated training time: $MAX_HOURS hours"
fi

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

echo "===== STARTING TRAINING ====="
# Verify DeepSpeed setup before training
verify_deepspeed_setup

# Train with direct module call (bypassing main_api.py)
echo "Starting training with direct module call (avoids argument mismatch)..."
python -m src.training.train \
    --config config/training_config.json \
    --data_dir data/processed \
    $USE_DRIVE_FLAG \
    --push_to_hub \
    --deepspeed \
    --deepspeed_config "$DS_CONFIG_PATH" \
    2>&1 | tee logs/train_a6000_optimized_$(date +%Y%m%d_%H%M%S).log

# Check exit status
EXIT_STATUS=$?

# Check for DeepSpeed-related errors in the log
LATEST_LOG=$(find logs -name "train_a6000_optimized_*.log" -type f -exec ls -t {} \; | head -1)
if [ -f "$LATEST_LOG" ]; then
    # Check for common DeepSpeed errors
    if grep -q "NoneType.* has no attribute 'hf_ds_config'" "$LATEST_LOG" || \
       grep -q "DeepSpeed configuration issue" "$LATEST_LOG" || \
       grep -q "DeepSpeedPlugin" "$LATEST_LOG" && grep -q "Error" "$LATEST_LOG"; then
        
        echo "===== DEEPSPEED ERROR DETECTED ====="
        echo "Running diagnostics script..."
        bash scripts/diagnose_deepspeed.sh 2>&1 | tee logs/deepspeed_error_$(date +%Y%m%d_%H%M%S).log
        
        echo "Attempting to fix DeepSpeed configuration and retry training..."
        # Fix DeepSpeed configuration
        python scripts/fix_deepspeed.py
        
        # Save the original log
        cp "$LATEST_LOG" "${LATEST_LOG%.log}_failed.log"
        
        echo "Retrying training with fixed configuration..."
        verify_deepspeed_setup
        python -m src.training.train \
            --config config/training_config.json \
            --data_dir data/processed \
            $USE_DRIVE_FLAG \
            --push_to_hub \
            --deepspeed \
            --deepspeed_config "$DS_CONFIG_PATH" \
            --debug \
            2>&1 | tee logs/train_a6000_optimized_retry_$(date +%Y%m%d_%H%M%S).log
            
        # Update exit status based on retry
        EXIT_STATUS=$?
    fi
fi

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