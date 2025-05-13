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

# Set drive options
if [ "$USE_DRIVE" = true ]; then
  echo "Setting up Google Drive authentication..."
  python scripts/setup_google_drive.py
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

# Step 1: Process datasets first
echo "==== Processing text datasets with optimizations ===="
python -m src.data.process_datasets \
  --config config/dataset_config_text.json \
  --datasets openassistant gpt_teacher pile_filtered \
  --streaming \
  --no_cache \
  $DRIVE_OPTS

# Step 2: Train the model with checkpointing and logging to Drive
echo "==== Training FLAN-UL2 with optimizations and Drive integration ===="
python train_text_flan.py \
  --config config/training_config_text.json \
  --data_dir data/processed \
  --push_to_hub \
  $DRIVE_OPTS \
  --debug \
  2>&1 | tee logs/train_flan_ul2_a6000_$(date +%Y%m%d_%H%M%S).log

# Step 3: Sync any remaining results to Drive if enabled
if [ "$USE_DRIVE" = true ]; then
  echo "==== Syncing all results to Google Drive ===="
  python scripts/sync_to_drive.py --sync-all \
    --base-dir "$DRIVE_BASE_DIR" \
    --is-text-model \
    $([ "$SKIP_LOCAL" = true ] && echo "--delete-local")
fi

echo "==== Training completed! ===="
echo "Check the text_models directory for model files or the Google Drive folder if Drive integration was enabled." 