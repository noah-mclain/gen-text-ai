#!/bin/bash
# Optimized training script for text generation with FLAN-UL2 on A6000 GPU
# This script includes Google Drive integration for cloud storage of datasets and models

# Set bash to exit on error
set -e

# Parse arguments
DRIVE_BASE_DIR="FlanUL2Text"
USE_DRIVE=true
SKIP_LOCAL=false
BATCH_SIZE=1
GRAD_ACCUMULATION=16
LR=1e-5
MAX_STEPS=10000
SAVE_STEPS=1000
WARMUP_STEPS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-drive)
      USE_DRIVE=false
      shift
      ;;
    --delete-local)
      SKIP_LOCAL=true
      shift
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --grad-accum)
      GRAD_ACCUMULATION="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --save-steps)
      SAVE_STEPS="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. Some datasets might be inaccessible and you won't be able to push to Hugging Face Hub."
  echo "You should set it using: export HF_TOKEN=your_huggingface_token"
else
  echo "HF_TOKEN is set. Will use for authentication with Hugging Face Hub."
fi

# Check Paperspace environment
echo "===== CHECKING PAPERSPACE ENVIRONMENT ====="
python scripts/check_paperspace_env.py

# Set CUDA visible devices to control which GPU is used
export CUDA_VISIBLE_DEVICES=0

# Set environment variables for better performance
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Make directories
mkdir -p logs
mkdir -p data/processed
mkdir -p text_models/flan-ul2-fine-tuned

# Clean any DeepSpeed environment variables that might cause conflicts
echo "===== CLEANING DEEPSPEED ENVIRONMENT VARIABLES ====="
chmod +x scripts/clean_deepspeed_env.py
python scripts/clean_deepspeed_env.py

# Unset any DeepSpeed variables in the current shell
unset ACCELERATE_USE_DEEPSPEED
unset ACCELERATE_DEEPSPEED_CONFIG_FILE
unset ACCELERATE_DEEPSPEED_PLUGIN_TYPE
unset HF_DS_CONFIG
unset DEEPSPEED_CONFIG_FILE
unset DS_ACCELERATOR
unset DS_OFFLOAD_PARAM
unset DS_OFFLOAD_OPTIMIZER

# Set up Google Drive environment
mkdir -p text_models/flan-ul2-fine-tuned

# Set up Google Drive authentication
if [ "${USE_DRIVE:-True}" = "True" ]; then
  echo "Setting up Google Drive authentication..."
  
  # Set text model-specific drive base directory
  export DRIVE_BASE_DIR="${DRIVE_BASE_DIR:-FlanUL2Text}"
  echo "Using text-specific Drive base directory: $DRIVE_BASE_DIR"
  
  python scripts/setup_google_drive.py --base_dir "$DRIVE_BASE_DIR"
  if [ $? -ne 0 ]; then
    echo "Google Drive authentication failed. Proceeding without Drive integration."
    USE_DRIVE="False"
    DRIVE_OPTS=""
  else
    echo "Google Drive authentication successful. Will use Drive for storage."
    DRIVE_OPTS="--use_drive --drive_base_dir $DRIVE_BASE_DIR"
    if [ "$SKIP_LOCAL" = true ]; then
      DRIVE_OPTS="$DRIVE_OPTS --skip_local_storage"
    fi
    
    # Validate authentication by testing if we can access the drive
    python -c "
import sys
from src.utils.google_drive_manager import test_drive_mounting
if not test_drive_mounting():
    print('Failed to access Google Drive after authentication')
    sys.exit(1)
else:
    print('Successfully validated Google Drive access')
" 
    
    # If validation failed, disable Drive
    if [ $? -ne 0 ]; then
      echo "Google Drive access validation failed. Proceeding without Drive integration."
      USE_DRIVE="False"
      DRIVE_OPTS=""
    fi
  fi
else
  USE_DRIVE="False"
  DRIVE_OPTS=""
  echo "Google Drive integration disabled"
fi

# Step 1: Get dataset names from the config file
echo "===== IDENTIFYING TEXT DATASETS FROM CONFIG ====="
TEXT_DATASETS=$(python -c "
import json
import sys
try:
    with open('config/dataset_config_text.json', 'r') as f:
        config = json.load(f)
        # Filter only enabled datasets
        enabled_datasets = [name for name, info in config.items() 
                           if info.get('enabled', True)]
        if enabled_datasets:
            print(' '.join(enabled_datasets))
        else:
            # Default datasets if none are enabled
            print('openassistant gpteacher_general pile synthetic_persona writingprompts')
except Exception as e:
    print('openassistant gpteacher_general pile')  # Fallback list
    sys.stderr.write(f'Error reading dataset config: {e}\\n')
")
echo "Datasets in configuration: $TEXT_DATASETS"

# Step 2: Check which datasets are already processed and available on Drive
echo "===== CHECKING FOR PRE-PROCESSED DATASETS ====="
if [ "${USE_DRIVE}" = "True" ]; then
    echo "Google Drive integration is ENABLED. Looking for pre-processed datasets in Drive folder: $DRIVE_BASE_DIR"
else
    echo "Google Drive integration is DISABLED. Using local datasets only."
fi

# Run the dataset checker to identify available vs needed datasets
echo "Running dataset availability check..."
DATASET_STATUS=$(python -c "
from src.utils.drive_dataset_checker import prepare_datasets
import json
import sys
import os

config_path = 'config/dataset_config_text.json'
skip_drive = ${USE_DRIVE} != 'True'
drive_folder = '${DRIVE_BASE_DIR}/preprocessed' if not skip_drive else None

try:
    # This gets datasets available locally or on Drive, and those still needed
    available, needed, download_time = prepare_datasets(
        config_path, 
        output_dir='data/processed',
        drive_folder=drive_folder,
        skip_drive=skip_drive
    )
    
    print('AVAILABLE=' + ','.join(available), file=sys.stdout)
    print('NEEDED=' + ','.join(needed), file=sys.stdout)
    print('DOWNLOAD_TIME=' + str(download_time), file=sys.stdout)
except Exception as e:
    print(f'Error checking datasets: {e}', file=sys.stderr)
    # Fallback to processing all datasets
    with open(config_path, 'r') as f:
        config = json.load(f)
        all_datasets = [name for name, info in config.items() if info.get('enabled', True)]
    print('AVAILABLE=', file=sys.stdout)
    print('NEEDED=' + ','.join(all_datasets), file=sys.stdout)
    print('DOWNLOAD_TIME=0', file=sys.stdout)
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

# Step 3: Process only the datasets that need processing
if [ -n "$DATASETS_TO_PROCESS" ]; then
    echo "===== PROCESSING DATASETS ====="
    echo "Processing datasets: $DATASETS_TO_PROCESS"
    
    python -m src.data.process_datasets \
      --config config/dataset_config_text.json \
      --datasets $DATASETS_TO_PROCESS \
      --streaming \
      --no_cache \
      $DRIVE_OPTS \
      --verbose
      
    if [ $? -ne 0 ]; then
        echo "⚠️ Dataset processing encountered errors. Training may use incomplete data."
    else  
        echo "✅ Dataset processing completed successfully."
    fi
else
    echo "✅ All datasets already available. Skipping dataset processing step."
fi

# Step 4: Train the model with checkpointing and logging to Drive
echo "==== Training FLAN-UL2 with optimizations and Drive integration ===="
python train_text_flan.py \
  --config config/training_config_text.json \
  --data_dir data/processed \
  --push_to_hub \
  --no_deepspeed \
  $DRIVE_OPTS \
  --debug \
  2>&1 | tee logs/train_flan_ul2_a6000_$(date +%Y%m%d_%H%M%S).log

# Step 5: Sync any remaining results to Drive if enabled
if [ "$USE_DRIVE" = true ]; then
  echo "==== Syncing all results to Google Drive ===="
  python scripts/sync_to_drive.py --sync-all \
    --base-dir "$DRIVE_BASE_DIR" \
    --is-text-model \
    $([ "$SKIP_LOCAL" = true ] && echo "--delete-local")
fi

echo "==== Training completed! ===="
echo "Check the text_models directory for model files or the Google Drive folder if Drive integration was enabled." 