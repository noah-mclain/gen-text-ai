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

# Make directories
mkdir -p logs
mkdir -p data/processed
mkdir -p text_models/flan-ul2-fine-tuned

# Set up Google Drive authentication
if [ "$USE_DRIVE" = true ]; then
  echo "Setting up Google Drive authentication..."
  
  # Set text model-specific drive base directory
  export DRIVE_BASE_DIR="${DRIVE_BASE_DIR:-FlanUL2Text}"
  echo "Using text-specific Drive base directory: $DRIVE_BASE_DIR"
  
  python scripts/setup_google_drive.py --base_dir "$DRIVE_BASE_DIR"
  if [ $? -ne 0 ]; then
    echo "Google Drive authentication failed. Proceeding without Drive integration."
    USE_DRIVE=false
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
      USE_DRIVE=false
      DRIVE_OPTS=""
    fi
  fi
else
  DRIVE_OPTS=""
  echo "Google Drive integration disabled"
fi

# Print training configuration
echo "==== Training Configuration ===="
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUMULATION"
echo "Learning rate: $LR"
echo "Max steps: $MAX_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
if [ "$USE_DRIVE" = true ]; then
  echo "Google Drive: Enabled (Base dir: $DRIVE_BASE_DIR)"
  if [ "$SKIP_LOCAL" = true ]; then
    echo "Local storage: Skip after backup to Drive"
  else
    echo "Local storage: Keep local copies"
  fi
else
  echo "Google Drive: Disabled"
fi

# Step 1: Get dataset names from the config file
echo "==== Getting text dataset names from config ===="
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

# Step 2: Process datasets with all the enabled ones from config
echo "==== Processing text datasets with optimizations ===="
echo "Datasets to process: $TEXT_DATASETS"
python -m src.data.process_datasets \
  --config config/dataset_config_text.json \
  --datasets $TEXT_DATASETS \
  --streaming \
  --no_cache \
  $DRIVE_OPTS

# Fix DeepSpeed configuration before training
echo "==== Setting up DeepSpeed configuration ===="
# Run the DeepSpeed fix script
python scripts/fix_deepspeed.py

# Set DeepSpeed environment variables
export DS_ACCELERATOR=cuda
export DS_OFFLOAD_PARAM=cpu
export DS_OFFLOAD_OPTIMIZER=cpu
export ACCELERATE_USE_DEEPSPEED=true
# Set explicit path to ensure accelerate can find it
export ACCELERATE_DEEPSPEED_CONFIG_FILE=$(pwd)/ds_config_a6000.json
echo "DeepSpeed config at: $ACCELERATE_DEEPSPEED_CONFIG_FILE"

# Step 3: Train the model with checkpointing and logging to Drive
echo "==== Training FLAN-UL2 with optimizations and Drive integration ===="
python train_text_flan.py \
  --config config/training_config_text.json \
  --data_dir data/processed \
  --push_to_hub \
  $DRIVE_OPTS \
  --debug \
  2>&1 | tee logs/train_flan_ul2_a6000_$(date +%Y%m%d_%H%M%S).log

# Step 4: Sync any remaining results to Drive if enabled
if [ "$USE_DRIVE" = true ]; then
  echo "==== Syncing all results to Google Drive ===="
  python scripts/sync_to_drive.py --sync-all \
    --base-dir "$DRIVE_BASE_DIR" \
    --is-text-model \
    $([ "$SKIP_LOCAL" = true ] && echo "--delete-local")
fi

echo "==== Training completed! ===="
echo "Check the text_models directory for model files or the Google Drive folder if Drive integration was enabled." 