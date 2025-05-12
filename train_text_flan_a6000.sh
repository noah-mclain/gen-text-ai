#!/bin/bash
# Optimized training script for text generation with FLAN-UL2 on A6000 GPU
# This version includes Google Drive integration for cloud storage of datasets and models

# Set bash to exit on error
set -e

# Parse arguments
DRIVE_BASE_DIR="DeepseekCoder"
USE_DRIVE=true
SKIP_LOCAL=true
USE_RCLONE=true
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
    --keep-local)
      SKIP_LOCAL=false
      shift
      ;;
    --drive-dir)
      DRIVE_BASE_DIR="$2"
      shift 2
      ;;
    --use-api)
      USE_RCLONE=false
      shift
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

# Set drive options
if [ "$USE_DRIVE" = true ]; then
  DRIVE_OPTS="--use_drive --drive_base_dir $DRIVE_BASE_DIR"
  if [ "$USE_RCLONE" = false ]; then
    DRIVE_OPTS="$DRIVE_OPTS --use_drive_api"
  fi
  if [ "$SKIP_LOCAL" = true ]; then
    DRIVE_OPTS="$DRIVE_OPTS --skip_local_storage"
  fi
  echo "Google Drive integration enabled with base directory: $DRIVE_BASE_DIR"
  if [ "$USE_RCLONE" = true ]; then
    echo "Using rclone for Google Drive operations"
  else
    echo "Using Google Drive API for operations"
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
  --training_config config/training_config_text.json \
  --model_name "google/flan-ul2" \
  --output_dir "text_models/flan-ul2-fine-tuned" \
  --dataset_config config/dataset_config_text.json \
  --datasets openassistant gpt_teacher pile_filtered \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --learning_rate $LR \
  --max_steps $MAX_STEPS \
  --save_steps $SAVE_STEPS \
  --warmup_steps $WARMUP_STEPS \
  --use_8bit_optimizer \
  --use_4bit_quantization \
  --use_lora \
  --use_deepspeed ZeRO-3 \
  --use_drive $USE_DRIVE \
  --drive_base_dir "$DRIVE_BASE_DIR" \
  --use_rclone $USE_RCLONE \
  --skip_local_storage $SKIP_LOCAL

# Step 3: Sync any remaining results to Drive if enabled
if [ "$USE_DRIVE" = true ]; then
  echo "==== Syncing all results to Google Drive ===="
  python scripts/sync_to_drive.py --sync-all \
    --base-dir "$DRIVE_BASE_DIR" \
    $([ "$USE_RCLONE" = false ] && echo "--use-api") \
    --is-text-model \
    $([ "$SKIP_LOCAL" = true ] && echo "--delete-local")
fi

echo "==== Training completed! ===="
echo "Check the text_models directory for model files or the Google Drive folder if Drive integration was enabled." 